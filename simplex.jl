using DataFrames
using Crayons
using LinearAlgebra

LOG_RESULTS = true # turn off if you don't want to see detailed logs

INFEASIBLE = :INFEASIBLE
FEASIBLE = :FEASIBLE
UNBOUNDED = :UNBOUNDED
header_text = Crayon(foreground=:green, bold=true, underline=true) # Bold green text
final_answer_text = Crayon(foreground=:red, bold=true) # Bold green text

function log(args...)
    if (LOG_RESULTS)
        println(join(args, ""))
        print(Crayon(reset=true))
    end
end

function disp(arg)
    if (LOG_RESULTS)
        display(arg)
    end
end

function simplex(; A::Matrix, b::Vector, c::Vector)
    phase1_result = phase1(A=A, b=b, c=c)
    log("phase I result: ", phase1_result)

    if (phase1_result[:status] == INFEASIBLE)
        return INFEASIBLE
    end

    return phase2(
        A=A,
        b=b,
        c=c,
        basis=phase1_result[:basis],
        A_B_inv=phase1_result[:A_B_inv],
    )
end

function phase1(; A::Matrix, b::Vector, c::Vector)
    log(header_text, "STARTING SIMPLEX PHASE I")

    A, basis, artificial_variable_columns = add_artificial_variable_columns_if_necessary(A)
    if (length(artificial_variable_columns) > 0)
        # must excise the artificial variables

        c = zeros(size(A, 2))
        for a in artificial_variable_columns
            c[a] = 1
        end
        log("artificial_variable_columns: ", artificial_variable_columns)
        log("A: ", A)
        log("basis: ", basis)
        log("c: ", c)
        # permutation matrix is a square binary matrix that has exactly one entry of 1 in each row and each column with all other entries 0
        A_B_inv = A[:, basis] # for a square "permutation matrix", its inverse is itself
        result = phase2(A=A, b=b, c=c, basis=basis, A_B_inv=A_B_inv)
        log("result: ", result)
        x, _, basis, A_B_inv = result
        log("basis: ", basis)
        log("A_B_inv: ", A_B_inv)
        still_contains_artificial = any(x -> x in artificial_variable_columns, basis)

        if (still_contains_artificial)
            log(final_answer_text, "STOP. Problem is infeasible.")
            return Dict(:status => INFEASIBLE)
        end

        return Dict(
            :status => FEASIBLE,
            :basis => basis,
            :A_B_inv => A_B_inv
        )
    end
    # permutation matrix is a square binary matrix that has exactly one entry of 1 in each row and each column with all other entries 0
    A_B_inv = A[:, basis] # for a square "permutation matrix", its inverse is itself
    return Dict(
        :status => FEASIBLE,
        :basis => basis,
        :A_B_inv => A_B_inv
    )
end

# find identity structure, else create it
function add_artificial_variable_columns_if_necessary(A::Matrix)::Tuple{Matrix,Vector,Vector}
    num_rows = size(A, 1)
    artificial_variable_columns = []

    # ith element of this vector has value k, meaning 
    # the ith row has the identity structure with a 1 in column k
    identity_structure_in_col = zeros(num_rows)

    for (col_idx, col) in enumerate(eachcol(A))
        # all elements are 0 except one is 1
        contains_identity_structure = count(x -> x == 1, col) == 1 && all(x -> x in (0, 1), col)
        if !contains_identity_structure
            continue
        end
        row_containing_1 = findfirst(v -> v == 1, col)
        identity_structure_in_col[row_containing_1] = col_idx
        if all(x -> x != 0, identity_structure_in_col)
            basis = Int.(identity_structure_in_col)
            return (A, basis, artificial_variable_columns)
        end
    end

    # for every row that doesn't have an identity structure, add one to the end
    basis = []
    for (row_idx, col_with_1) in enumerate(identity_structure_in_col)
        if (col_with_1 == 0)
            new_col = zeros(num_rows)
            new_col[row_idx] = 1
            A = hcat(A, new_col)
            num_cols = size(A, 2)
            push!(basis, num_cols)
            push!(artificial_variable_columns, num_cols)
        else
            push!(basis, col_with_1)
        end
    end

    return (A, Int.(basis), artificial_variable_columns)
end


function phase2(; A::Matrix, b::Vector, c::Vector, basis::Vector, A_B_inv::Union{Matrix,Diagonal})
    log(header_text, "STARTING SIMPLEX PHASE II")
    log("starting basis: $(basis)")
    iteration = 1
    while (true)
        log(Crayon(foreground=:yellow), "iteration: $iteration")
        # compute basic solution
        log("A_B_inv_here")
        display(A_B_inv)
        b_bar = A_B_inv * b
        log("b_bar= $b_bar")

        # Now compute the duals
        c_B = c[basis]
        log("c_B", c_B)
        y_bar = c_B' * A_B_inv
        log("y_bar", y_bar)
        c_bar = c' - y_bar * A
        log("c_bar", c_bar)



        # index of incoming variable to basis
        # Bland's rule, when choosing incoming variable, choose one with smallest index
        t = findfirst(v -> v < 0, c_bar') # could also do find last
        log("t=", t)
        if (t === nothing)
            log(final_answer_text, "STOP, optimum found. basis is ", basis)
            log("b_bar=$(b_bar)")
            values = printresults(A, basis, b_bar)
            obj_value = c_B' * b_bar
            log("objective value is", obj_value)
            return (values, obj_value, basis, A_B_inv)
        end

        A_bar_◎t = A_B_inv * A[:, t]
        log("A_bar_◎t", A_bar_◎t)
        # if no positive entries, we are unbounded
        if findfirst(v -> v > 0, A_bar_◎t) === nothing
            log(final_answer_text, "STOP, unbounded.")
            return UNBOUNDED
        end

        # minimum ratio test
        log("MIN-RATIO TEST")
        ✽ = 1 / typemax(Int32) # kinda hacky, just want a huge value
        A_bar_◎t_modified = map(v -> v < 0 ? ✽ : v, A_bar_◎t)
        log("A_bar_◎t_modified", A_bar_◎t_modified)
        args = [b_bar[i] / A_bar_◎t_modified[i] for i ∈ 1:length(b_bar)]

        # Bland's rule here also, in the case of multiple equal and minimal values,
        # argmin returns the index of the first one
        r = argmin(args) # position to be excised from basis
        log("args=", args)
        log("r=$r is the position to be excised from the basis.")
        basis[r] = t
        log("new basis", basis)

        # calculate permutation matrix
        P = Matrix{Float64}(LinearAlgebra.I(length(basis)))
        rth_col = ones(length(basis))
        for i in eachindex(rth_col)
            num = (r == i) ? 1 : -A_bar_◎t[i]
            den = A_bar_◎t[r]
            rth_col[i] = (num / den)
        end

        log("rth col= ", rth_col)
        P[:, r] = rth_col
        log("P")
        disp(P)
        log()


        A_B_inv = P * A_B_inv
        log("new A_B_inv")
        disp(A_B_inv)
        log()

        iteration += 1
    end
end


# print out our x values
function printresults(A, basis, b_bar)
    variables = ["x$col" for col ∈ 1:size(A, 2)]
    values = zeros(size(variables))
    for i ∈ eachindex(variables)
        b_bar_ind = findfirst(v -> v == i, basis)
        if b_bar_ind !== nothing
            values[i] = b_bar[b_bar_ind]
        end
    end
    df = DataFrame(variables=variables, values=values)
    show(df)
    log()
    return values
end