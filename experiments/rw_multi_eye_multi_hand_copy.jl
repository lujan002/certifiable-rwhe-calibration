using DelimitedFiles
using Rotations
using SparseArrays
using MKL
using Plots
using LinearAlgebra

include("../src/calibration/robot_world_costs.jl");
include("../src/rotation_sdp_solver_jump.jl");

function quat_trans2poses(quat_tran_mtx, invert=false)
    #Ensure the matrix has at least 2 dimensions
    if ndims(quat_tran_mtx) == 1
        quat_tran_mtx = reshape(quat_tran_mtx, (1,7))
    end

    poses = zeros(size(quat_tran_mtx, 1),4,4)
    poses[:, 4, 4] .=1
    for (index,row) in enumerate(eachrow(quat_tran_mtx))
        if invert
            poses[index, 1:3, 1:4] = hcat(transpose(QuatRotation(row[1:4])), -transpose(QuatRotation(row[1:4])) * reshape(row[5:7], (3, 1)))
        else
            poses[index, 1:3, 1:4] = hcat(QuatRotation(row[1:4]), reshape(row[5:7], (3, 1)))
        end
    end
    return poses
end

function evaluate_Y(A, B, X, n)
    Y = zeros(n, 4, 4)
    for i=1:n
        Y[i, :, :] = A[i, :, :]*X*inv(B[i, :, :])
    end
    show(stdout, "text/plain", Y[:, 1:3, 4])
    show(stdout, "text/plain", Y[:, 1:3, 4])
end

function evaluate_err(A, B, X, Y, n)
    err = zeros(n, 2)
    for i=1:n
        pose_err = A[i, :, :]*X - Y*B[i, :, :]
        err[i, 1] = norm(pose_err[1:3,1:3])
        err[i, 2] = norm(pose_err[1:3, 4])
    end
    display(err)
end

function transform_poses_by_ref(poses, A_ref_inv)
    """Transform all poses by A_ref_inv to anchor world frame to the reference pose.
    
    Given poses A_i representing world -> optitrack_i, transform to A'_i = A_ref_inv * A_i
    so that A'_ref = I (identity), anchoring the world frame to the reference optitrack pose.
    """
    n = size(poses, 1)
    transformed = zeros(n, 4, 4)
    for i in 1:n
        transformed[i, :, :] = A_ref_inv * poses[i, :, :]
    end
    return transformed
end

function load_reference_A_pose(folder, dataset)
    """Load the first A pose from the first A file to use as reference for anchoring world frame."""
    directory_list = readdir(join([pwd(), "/data", folder, dataset]))
    iter = Iterators.filter(item -> collect(split(item, ".")) != 1, directory_list)
    iter_A = sort(collect(Iterators.filter(item -> last(split(split(item, '.')[1], '_')) == "A", collect(iter))))
    
    if isempty(iter_A)
        error("No A files found in dataset")
    end
    
    # Load first A file and get first pose
    first_A_file = iter_A[1]
    println("Using reference A pose from: $(first_A_file)")
    A_quat_trans = readdlm(join([pwd(), "/data", folder, dataset, "/", first_A_file]), ',', Float64, '\n')
    A_poses = quat_trans2poses(A_quat_trans)
    
    A_ref = A_poses[1, :, :]
    A_ref_inv = inv(A_ref)
    
    println("Reference A pose (first measurement from first file):")
    display(A_ref)
    
    return A_ref_inv
end

function process_data(folder, dataset)
    half_langevin_k = 125
    trans_std_dev = 0.1

    directory_list = readdir(join([pwd(), "/data", folder, dataset]))

    iter = Iterators.filter(item -> collect(split(item, ".")) !=1 , directory_list)
    iter_A = Iterators.filter(item -> last(split(split(item, '.')[1], '_')) == "A", collect(iter))
    iter_B = Iterators.filter(item -> last(split(split(item, '.')[1], '_')) == "B", collect(iter))
    iter_AB = Iterators.filter(pair -> (split(pair[1], "_")[2] == split(pair[2], "_")[2]) && (split(pair[1], "_")[4] == split(pair[2], "_")[4]), zip(iter_A, iter_B))
    tags = sort(unique([parse(Int64, split(pair[1], "_")[2]) for pair in iter_AB]))
    cameras = sort(unique([parse(Int64, split(pair[1], "_")[4]) for pair in iter_AB]))
    n = length(tags)
    m = length(cameras)
    println(join(["Tags Found:", tags]))
    println(join(["Cameras Found:", cameras]))
    println("Loading Data")
    Q = get_empty_sparse_cost_matrix(n, m, false)

    # Load reference A pose to anchor world frame to first OptiTrack measurement
    A_ref_inv = load_reference_A_pose(folder, dataset)
    println("World frame anchored to first OptiTrack measurement")

    for pair in iter_AB
        display(join([pwd(), "/data", folder, dataset, "/" , pair[1]]))
        
        A_quat_trans = readdlm(join([pwd(), "/data", folder, dataset, "/" , pair[1]]), ',', Float64, '\n')
        B_quat_trans = readdlm(join([pwd(), "/data", folder, dataset, "/" , pair[2]]), ',', Float64, '\n')
        num_meas = size(A_quat_trans, 1)
        
        #X_guess = [ -0.192639  -0.234029    0.952954   3.37587;
        #0.124476  -0.969126   -0.212838   0.431168;
        #0.973343   0.0776193   0.215823  -1.64415;
        #0.0        0.0         0.0        1.0]
        A_poses_raw = quat_trans2poses(A_quat_trans)
        B_poses = quat_trans2poses(B_quat_trans)
        
        # Transform A poses to anchor world frame to reference pose
        A_poses = transform_poses_by_ref(A_poses_raw, A_ref_inv)
        frame_x = parse(Int64, split(pair[1], "_")[2])
        frame_y = parse(Int64, split(pair[1], "_")[4])
        x_index = findfirst(item -> item == frame_x, tags)
        y_index = findfirst(item -> item == frame_y, cameras)
        #evaluate_Y(A_poses, B_poses, X_guess, num_meas)

        if num_meas < 3
            continue
        end

        #if num_meas > 2
        #    mask = rand(Bool,num_meas)
        #else
        #    mask = [true;true]
        #end
        #display(transpose(mask))
        #Q_temp = sparse_robot_world_transformation_cost(A_poses[mask, :, :], B_poses[mask, :, :], half_langevin_k .* ones(sum(mask)), (0.5/(trans_std_dev^2)) .* ones(sum(mask)))
        Q_temp = sparse_robot_world_transformation_cost(A_poses, B_poses, half_langevin_k .* ones(num_meas), (0.5/(trans_std_dev^2)) .* ones(num_meas))
        #Q_eigs, _ = eigen(Matrix(Q_temp))
        #println("Q_eigs for tag $(x_index) and cam $(y_index): $(Q_eigs)")
        #if frame_x==0
        # if (frame_x==2 || frame_x==4) && (frame_y==0 || frame_y==5)
        #     println("tag $(frame_x) and cam $(frame_y)")
        # Z_pair, model_pair = solve_sdp_dual_using_concat_schur(Q_temp, 2, true, true)
        # Q_eigs, _ = eigen(Matrix(Z_pair))
        # display(Q_eigs')
        # solution_pair = extract_solution_from_dual_schur(Z_pair, Q_temp);
        # X_temp = vcat(hcat(RotMatrix{3}(reshape(solution_pair[7:15], (3,3))), solution_pair[1:3]), [0 0 0 1])
        # Y_temp = vcat(hcat(RotMatrix{3}(reshape(solution_pair[16:24], (3,3))), solution_pair[4:6]), [0 0 0 1])
        # display(X_temp)
        # display(Y_temp)
        # display(X_temp[1:3, 1:3]'X_temp[1:3, 1:3])
        # display(Y_temp[1:3, 1:3]'Y_temp[1:3, 1:3])
        #end
        #evaluate_err(A_poses, B_poses, X_temp, Y_temp, size(A_poses, 1))
        Q += get_ith_eye_jth_base_cost(x_index, y_index, n, m, Q_temp, false)
    end
    #Q_eigs, _ = eigen(Matrix(Q))
    #println("Q_eigs for all: $(Q_eigs)")
    Z_schur, model = solve_sdp_dual_using_concat_schur(Q, n+m, true, true)
    #Q_eigs, _ = eigen(Matrix(Z_schur))
    #display(Q_eigs)
    #println(solution_summary(model))
    solution = extract_solution_from_dual_schur(Z_schur, Q)

    for i in range(1, length(tags) + length(cameras))
        display("Solution $(vcat(tags,cameras)[i]):")
        if i<=length(tags)
            display(reshape(solution[9*(i-1)+3*(n+m)+1:9*i+3*(n+m)], 3, 3))
            R_temp = reshape(solution[9*(i-1)+3*(n+m)+1:9*i+3*(n+m)], 3, 3)
            #display(R_temp'R_temp)
            display(solution[3*i-2:3*i])
        else
            display(reshape(solution[9*(i-1)+3*(n+m)+1:9*i+3*(n+m)], 3, 3))
            display(solution[3*i-2:3*i])
            #R_temp = reshape(solution[9*(i-1)+3*(n+m)+1:9*i+3*(n+m)], 3, 3)
            #R_0_temp = reshape(solution[9*(length(tags)+1-1)+3*(n+m)+1:9*(length(tags)+1)+3*(n+m)], 3, 3)
            #display(vcat(hcat(R_0_temp',- R_0_temp'solution[3*(length(tags)+1)-2:3*(length(tags)+1)]), [0 0 0 1]) * vcat(hcat(R_temp, solution[3*i-2:3*i]), [0 0 0 1]))
            #display(R_0_temp'solution[3*i-2:3*i] - R_0_temp'solution[3*(length(tags)+1)-2:3*(length(tags)+1)])
        end
    end

    #writedlm("filtered_results2.csv", solution_mtx, ',')
    # for i in range(1, length(tags)+length(cameras))
    #     solution[3*i-2:3*i]
    #     reshape(solution[9*(i-1)+3*(n+1)+1:9*i+3*(n+1)], 3, 3)
    # end
    save_poses_to_csv(tags, cameras, solution)
    return solution
end

function save_poses_to_csv(tags, cameras, solution)
    n_tags = length(tags)
    total = n_tags + length(cameras)
    
    if n_tags == 0
        println("No tag poses to save.")
        return
    end

    # Only extract tag poses (tags come first in the solution)
    translations = [solution[3*i-2:3*i] for i in 1:n_tags]
    rotations = [reshape(solution[9*(i-1)+3*(total)+1:9*i+3*(total)], 3, 3) for i in 1:n_tags]

    # Prepare data matrix: tag_id, tx, ty, tz, r11, r12, r13, r21, r22, r23, r31, r32, r33
    data = zeros(n_tags, 13)
    for i in 1:n_tags
        data[i, 1] = tags[i]  # tag_id
        data[i, 2:4] = translations[i]  # tx, ty, tz
        data[i, 5:13] = vec(rotations[i])  # rotation matrix flattened row-wise
    end

    # Save to CSV
    csv_filename = "tag_poses.csv"
    header = ["tag_id", "tx", "ty", "tz", "r11", "r12", "r13", "r21", "r22", "r23", "r31", "r32", "r33"]
    open(csv_filename, "w") do io
        println(io, join(header, ","))
        for i in 1:n_tags
            println(io, join([data[i, j] for j in 1:13], ","))
        end
    end
    
    println("Tag poses saved to: $(joinpath(pwd(), csv_filename))")
end


#folder = "/rw_processed_undistorted"
# folder = "/rw2_processed_2_subfolder"
# dataset = "/set1"
folder = "/high-bay"
dataset = "/combined"
solution = process_data(folder, dataset);
#solution_mtx = hcat(reshape(solution[3*(6)+1:12*(6)], 6, 9),reshape(solution[1:3*(6)], m+n, 3))