import taichi as ti
import numpy as np
import math
import sph_base


poly6_factor = 315.0 / 64.0 / math.pi
spiky_grad_factor = -45.0 / math.pi
time_delta = 1.0 / 20.0
dim = 3
particle_radius = 0.15
cell_size = 2.51
cell_recpr = 1.0 / cell_size
epsilon = 1e-5
max_num_particles_per_cell = 100
max_num_neighbors = 100

# PBF params
h = 1.1
mass = 1.0
rho0 = 1.0
lambda_epsilon = 100.0
pbf_num_iters = 5
corr_deltaQ_coeff = 0.3
corrK = 0.001
# Need ti.pow()
# corrN = 4.0
neighbor_radius = h * 1.05

def round_up(f, s):
    return (math.floor(f * cell_recpr / s) + 1) * s

@ti.data_oriented
class PBFSolver(sph_base.SPHBase):
    def __init__(self, particle_system):
        super().__init__(particle_system)
        self.gamma = self.ps.config['gamma']
        self.B = self.ps.config['B']
        self.surface_tension = ti.field(ti.f32, shape=())
        self.surface_tension[None] = self.ps.config['surfaceTension']

        self.domain_start = np.array(self.ps.config['domainStart'])
        self.domain_end = np.array(self.ps.config['domainEnd'])

        self.boundary = self.domain_end

        self.old_positions = ti.Vector.field(dim, float)
        self.grid_num_particles = ti.field(int)
        # grid2particles = ti.field(int)
        self.particle_num_neighbors = ti.field(int)
        self.particle_neighbors = ti.field(int)

        self.grid_size = (round_up(self.boundary[0], 1), round_up(self.boundary[1], 1), round_up(self.boundary[2], 1))

        print("self.grid_size: ", self.grid_size)
        print("self.boundary: ", self.boundary)

        self.grid2particles = ti.field(int, (self.grid_size + (max_num_particles_per_cell,)))
        self.lambdas = ti.field(float)
        self.position_deltas = ti.Vector.field(dim, float)

        # ti.root.dense(ti.i, self.ps.total_fluid_particle_num).place(self.old_positions)
        ti.root.dense(ti.i, self.ps.total_particle_num).place(self.old_positions)
        self.grid_snode = ti.root.dense(ti.ijk, self.grid_size) 
        self.grid_snode.place(self.grid_num_particles)
        # grid_snode.dense(ti.i, max_num_particles_per_cell).place(grid2particles) #this way cannot place a 4 dimension array
        self.grid2particles = ti.field(int, (self.grid_size + (max_num_particles_per_cell,)))
        # self.nb_node = ti.root.dense(ti.i, self.ps.total_fluid_particle_num)
        self.nb_node = ti.root.dense(ti.i, self.ps.total_particle_num)
        self.nb_node.place(self.particle_num_neighbors)
        self.nb_node.dense(ti.j, max_num_neighbors).place(self.particle_neighbors)
        # ti.root.dense(ti.i, self.ps.total_fluid_particle_num).place(self.lambdas, self.position_deltas)
        ti.root.dense(ti.i, self.ps.total_particle_num).place(self.lambdas, self.position_deltas)

    @ti.func
    def poly6_value(self, s, h):
        result = 0.0
        if 0 < s and s < h:
            x = (h * h - s * s) / (h * h * h)
            result = poly6_factor * x * x * x
        return result
    
    @ti.func
    def spiky_gradient(self, r, h):
        result = ti.Vector([0.0, 0.0, 0.0])
        r_len = r.norm()
        if 0 < r_len and r_len < h:
            x = (h - r_len) / (h * h * h)
            g_factor = spiky_grad_factor * x * x
            result = r * g_factor / r_len
        return result

    @ti.func
    def compute_scorr(self, pos_ji):
        # Eq (13)
        x = self.poly6_value(pos_ji.norm(), h) / self.poly6_value(corr_deltaQ_coeff * h, h)
        # pow(x, 4)
        x = x * x
        x = x * x
        return (-corrK) * x
    
    @ti.func
    def confine_position_to_boundary(self, p):
        bmin = particle_radius
        bmax = ti.Vector([self.boundary[0], self.boundary[1], self.boundary[2]
                        ]) - particle_radius
        for i in ti.static(range(dim)):
            # Use randomness to prevent particles from sticking into each other after clamping
            if p[i] <= bmin:
                p[i] = bmin + epsilon * ti.random()
            elif bmax[i] <= p[i]:
                p[i] = bmax[i] - epsilon * ti.random()
        return p


    @ti.func
    def get_cell(self, pos):
        return int(pos * cell_recpr)
    
    @ti.func
    def is_in_grid(self, c):
        # @c: Vector(i32)
        return 0 <= c[0] and c[0] < self.grid_size[0] and 0 <= c[1] and c[
            1] < self.grid_size[1] and c[2] >= 0 and c[2] < self.grid_size[2] 

    @ti.kernel
    def prologue(self):
        # save old positions
        for i in self.ps.position:
            if self.ps.material[i] == self.ps.material_fluid:
                self.old_positions[i] = self.ps.position[i]
        # apply gravity within boundary
        for i in self.ps.position:
            if self.ps.material[i] == self.ps.material_fluid:
                g = ti.Vector([0.0, -9.81, 0.0])
                pos, vel = self.ps.position[i], self.ps.velocity[i]
                # print("pos_before",i, pos)
                vel += g * time_delta
                # print("vel",i, vel)
                pos += vel * time_delta
                # print("pos_after",i, pos)
                self.ps.position[i] = self.confine_position_to_boundary(pos)
                # print("self.ps.position[",i,"]", self.ps.position[i])

        # clear neighbor lookup table
        for I in ti.grouped(self.grid_num_particles):
            # if self.ps.material[I] == self.ps.material_fluid:
            self.grid_num_particles[I] = 0
        for I in ti.grouped(self.particle_neighbors):
            # if self.ps.material[I] == self.ps.material_fluid:
            self.particle_neighbors[I] = -1

        # update grid
        for p_i in self.ps.position:
            # print(p_i, self.ps.material_fluid)
            if self.ps.material[p_i] == self.ps.material_fluid:
                cell = self.get_cell(self.ps.position[p_i])
                # ti.Vector doesn't seem to support unpacking yet
                # but we can directly use int Vectors as indices
                offs = ti.atomic_add(self.grid_num_particles[cell], 1)
                # print(p_i,"in cell", cell, "with offs", offs)
                self.grid2particles[cell, offs] = p_i
        # find particle neighbors
        for p_i in self.ps.position:
            if self.ps.material[p_i] == self.ps.material_fluid:
                pos_i = self.ps.position[p_i]
                cell = self.get_cell(pos_i)
                nb_i = 0
                for offs in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2),(-1, 2)))):
                    cell_to_check = cell + offs
                    if self.is_in_grid(cell_to_check):
                        for j in range(self.grid_num_particles[cell_to_check]):
                            p_j = self.grid2particles[cell_to_check, j]
                            if nb_i < max_num_neighbors and p_j != p_i and (
                                    pos_i - self.ps.position[p_j]).norm() < neighbor_radius:
                                self.particle_neighbors[p_i, nb_i] = p_j
                                nb_i += 1
                self.particle_num_neighbors[p_i] = nb_i

    
    @ti.kernel
    def substep_(self):
        # compute lambdas
        # Eq (8) ~ (11)
        # return
        for p_i in self.ps.position:
            pos_i = self.ps.position[p_i]
            # print("posi",p_i, pos_i)

            grad_i = ti.Vector([0.0, 0.0, 0.0])
            sum_gradient_sqr = 0.0
            density_constraint = 0.0

            # Check if the particle is fluid
            if self.ps.material[p_i] == self.ps.material_fluid:
                for j in range(self.particle_num_neighbors[p_i]):
                    p_j = self.particle_neighbors[p_i, j]
                    if p_j < 0:
                        break
                    pos_ji = pos_i - self.ps.position[p_j]
                    grad_j = self.spiky_gradient(pos_ji, h)
                    grad_i += grad_j
                    sum_gradient_sqr += grad_j.dot(grad_j)
                    # Eq(2)
                    density_constraint += self.poly6_value(pos_ji.norm(), h)

                # Eq(1) 
                density_constraint = (mass * density_constraint / rho0) - 1.0

                sum_gradient_sqr += grad_i.dot(grad_i)
                temp = (-density_constraint) / (sum_gradient_sqr + lambda_epsilon)
                self.lambdas[p_i] = temp
                # print("self.lambdas",p_i,self.lambdas[p_i])

        # compute position deltas
        # Eq(12), (14)
        for p_i in self.ps.position:
            pos_i = self.ps.position[p_i]
            lambda_i = self.lambdas[p_i]
            pos_delta_i = ti.Vector([0.0, 0.0, 0.0])

            if self.ps.material[p_i] == self.ps.material_fluid:
                for j in range(self.particle_num_neighbors[p_i]):
                    p_j = self.particle_neighbors[p_i, j]
                    if p_j < 0:
                        break

                    lambda_j = self.lambdas[p_j]
                    pos_ji = pos_i - self.ps.position[p_j]
                    scorr_ij = self.compute_scorr(pos_ji)
                    pos_delta_i += (lambda_i + lambda_j + scorr_ij) * \
                        self.spiky_gradient(pos_ji, h)

                pos_delta_i /= rho0
                self.position_deltas[p_i] = pos_delta_i
                # print("self.position_deltas[p_i]",p_i,self.position_deltas[p_i])
                # self.position_deltas[p_i] = ti.Vector([0.001, 0.0, 0.0])

        for i in self.ps.position:
            if self.ps.material[i] == self.ps.material_fluid:
                self.ps.position[i] += self.position_deltas[i]


    @ti.kernel
    def epilogue(self):
        # confine to boundary 
        test = self.ps.position[2]
        for i in self.ps.position:
            if self.ps.material[i] == self.ps.material_fluid:
                pos = self.ps.position[i]
                self.ps.position[i] = self.confine_position_to_boundary(pos)

        # update velocities
        for i in self.ps.position:
            if self.ps.material[i] == self.ps.material_fluid:
                self.ps.velocity[i] = (self.ps.position[i] - self.old_positions[i]) / time_delta
        # print(self.ps.velocity[1])
        # no vorticity/xsph because we cannot do cross product in 2D...

    def substep(self):
        self.prologue()
        for _ in range(pbf_num_iters):
            self.substep_()
        self.epilogue()



        