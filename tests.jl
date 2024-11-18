using Test
using LinearAlgebra

include("./simplex.jl")

# HW 8 problem 4 without given basis
@test simplex(
    A=[-1 1 1 0
        1 -1 0 1],
    b=[4, 10],
    c=[-4, -5, 0, 0]) == UNBOUNDED


@testset "add_artificial_variable_columns_if_necessary" begin
    A, basis, artificial_variable_columns = add_artificial_variable_columns_if_necessary([
        1 0 1
        0 1 1
    ])
    @test A == [1 0 1; 0 1 1]
    @test basis == [1, 2]
    @test artificial_variable_columns == []

    A, basis, artificial_variable_columns = add_artificial_variable_columns_if_necessary([
        0 1 1
        1 1 0
    ])
    @test A == [0 1 1; 1 1 0]
    @test basis == [3, 1]
    @test artificial_variable_columns == []


    A, basis, artificial_variable_columns = add_artificial_variable_columns_if_necessary([
        1 0 1
        1 1 1
    ])
    @test A == [1 0 1 1; 1 1 1 0]
    @test basis == [4, 2]
    @test artificial_variable_columns == [4]

    A, basis, artificial_variable_columns = add_artificial_variable_columns_if_necessary([
        1 1
        1 1
    ])
    @test A == [1 1 1 0; 1 1 0 1]
    @test basis == [3, 4]
    @test artificial_variable_columns == [3, 4]


    A, basis, artificial_variable_columns = add_artificial_variable_columns_if_necessary([
        1 1
        1 1
        0 0
        0 0
    ])
    @test A == [
        1 1 1 0 0 0
        1 1 0 1 0 0
        0 0 0 0 1 0
        0 0 0 0 0 1
    ]
    @test basis == [3, 4, 5, 6]
    @test artificial_variable_columns == [3, 4, 5, 6]

    A, basis, artificial_variable_columns = add_artificial_variable_columns_if_necessary([
        1 1 1
        1 0 1
    ])
    @test A == [1 1 1 0; 1 0 1 1]
    @test basis == [2, 4]
    @test artificial_variable_columns == [4]
end


# Homework 9 problem 4, infeasible
@test phase1(
    A=[-2 1 -1 0
        0 1 0 1
    ],
    b=[2, 1],
    c=[9, 1, 0, 0]
)[:status] == INFEASIBLE


# HW 7 problem 4
x, obj_value = phase2(
    A=[
        1 1 -1 0 0
        -1 1 0 -1 0
        0 1 0 0 1],
    b=[2, 1, 3],
    c=[1, -2, 0, 0, 0],
    basis=[1, 2, 5],
    A_B_inv=[ # from zork
        0.5 -0.5 0
        0.5 0.5 0
        -0.5 -0.5 1]
)
@test x == [0, 3, 1, 2, 0]
@test obj_value == -6

# HW 7 problem 4 without the given basis and A_B_inv
phase1_result = phase1(
    A=[
        1 1 -1 0 0
        -1 1 0 -1 0
        0 1 0 0 1],
    b=[2, 1, 3],
    c=[1, -2, 0, 0, 0]
)
@test phase1_result[:basis] == [1, 2, 5]
@test phase1_result[:A_B_inv] == [
    0.5 -0.5 0
    0.5 0.5 0
    -0.5 -0.5 1]

# HW 8 problem 4
@test phase2(
    A=[-1 1 1 0
        1 -1 0 1],
    b=[4, 10],
    c=[-4, -5, 0, 0],
    basis=[3, 4],
    A_B_inv=LinearAlgebra.I(2)
) == UNBOUNDED

# HW 8 problem 4 without given basis
@test simplex(
    A=[-1 1 1 0
        1 -1 0 1],
    b=[4, 10],
    c=[-4, -5, 0, 0]) == UNBOUNDED



# 3D Klee Minty Problem
x, obj_value = simplex(
    A=[
        1 0 0 1 0 0
        2 1 0 0 1 0
        4 2 1 0 0 1
    ],
    b=[1, 2, 4],
    c=[-1, -2, -4, 0, 0, 0]
)
@test x[3] == 4