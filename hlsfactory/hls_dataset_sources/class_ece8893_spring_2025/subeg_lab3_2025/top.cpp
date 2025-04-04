#include "dcl.h"
#include "hls_stream.h"


void copy_A_to_local(
    data_t values_A_global[N * M],
    int    column_indices_A_global[N * M],
    int    row_ptr_A_global[N + 1],
    data_t values_A_local[N * M],
    int    column_indices_A_local[N * M],
    int    row_ptr_A_local[N + 1]
) {

    copy_row_ptr_A: for (int i = 0; i < N + 1; i++) {
    #pragma HLS PIPELINE II=1
        row_ptr_A_local[i] = row_ptr_A_global[i];
    }

    int nnz_A = row_ptr_A_global[N];

    copy_A: for (int i = 0; i < nnz_A; i++) {
    #pragma HLS PIPELINE II=1
        values_A_local[i]         = values_A_global[i];
        column_indices_A_local[i] = column_indices_A_global[i];
    }
}


void copy_B_to_local(
    data_t values_B_global[M * K],
    int    row_indices_B_global[M * K],
    int    col_ptr_B_global[M + 1],
    data_t values_B_local[M * K],
    int    row_indices_B_local[M * K],
    int    col_ptr_B_local[M + 1]
) {

    copy_col_ptr_B: for (int i = 0; i < M + 1; i++) {
    #pragma HLS PIPELINE II=1
        col_ptr_B_local[i] = col_ptr_B_global[i];
    }

    int nnz_B = col_ptr_B_global[M];

    copy_B: for (int i = 0; i < nnz_B; i++) {
    #pragma HLS PIPELINE II=1
        values_B_local[i]      = values_B_global[i];
        row_indices_B_local[i] = row_indices_B_global[i];
    }
}


void stage1_function(
    int i,
    int row_ptr_A_local[N + 1],
    int column_indices_A_local[N * M],
    data_t values_A_local[N * M],
    hls::stream<int> &s_k,
    hls::stream<data_t> &s_valA
) {
    stage1_loop: for (int idx = row_ptr_A_local[i]; idx < row_ptr_A_local[i + 1]; idx++) {
    #pragma HLS PIPELINE II=1
        int k = column_indices_A_local[idx];
        data_t a_val = values_A_local[idx];
        s_k.write(k);
        s_valA.write(a_val);
    }
}


void stage2_function(
    int i,
    int row_ptr_A_local[N + 1],
    int column_indices_A_local[N * M],
    data_t values_A_local[N * M],
    int col_ptr_B_local[M + 1],
    int row_indices_B_local[M * K],
    data_t values_B_local[M * K],
    hls::stream<int> &s_k,
    hls::stream<data_t> &s_valA,
    hls::stream<int> &s_j,
    hls::stream<data_t> &s_valB,
    hls::stream<data_t> &s_a_val
) {
    // Number of nonzeros in row i of A.
    int numA = row_ptr_A_local[i + 1] - row_ptr_A_local[i];
    stage2_loop: for (int idx = 0; idx < numA; idx++) {
    #pragma HLS PIPELINE II=1
        int k = s_k.read();
        data_t a_val = s_valA.read();
        // For each nonzero in column k of B.
        for (int idxB = col_ptr_B_local[k]; idxB < col_ptr_B_local[k + 1]; idxB++) {
        #pragma HLS PIPELINE II=1
            int j = row_indices_B_local[idxB];
            data_t b_val = values_B_local[idxB];
            s_j.write(j);
            s_valB.write(b_val);
            s_a_val.write(a_val);
        }
    }
}


void stage3_function(
    int i,
    int row_ptr_A_local[N + 1],
    int column_indices_A_local[N * M],
    data_t values_A_local[N * M],
    int col_ptr_B_local[M + 1],
    int row_indices_B_local[M * K],
    data_t values_B_local[M * K],
    hls::stream<int> &s_j,
    hls::stream<data_t> &s_valB,
    hls::stream<data_t> &s_a_val,
    data_t local_C_row[K]
) {

    int count = 0;
    count_loop: for (int idx = row_ptr_A_local[i]; idx < row_ptr_A_local[i + 1]; idx++) {
    #pragma HLS PIPELINE II=1
        int k = column_indices_A_local[idx];
        count += (col_ptr_B_local[k + 1] - col_ptr_B_local[k]);
    }
    stage3_loop: for (int idx = 0; idx < count; idx++) {
    #pragma HLS PIPELINE II=1
        int j = s_j.read();
        data_t b_val = s_valB.read();
        data_t a_val = s_a_val.read();
        local_C_row[j] += a_val * b_val;
    }
}


void process_row(
    int i,
    int row_ptr_A_local[N + 1],
    int column_indices_A_local[N * M],
    data_t values_A_local[N * M],
    int col_ptr_B_local[M + 1],
    int row_indices_B_local[M * K],
    data_t values_B_local[M * K],
    data_t local_C_row[K]
) {

    init_row: for (int j = 0; j < K; j++) {
    #pragma HLS PIPELINE II=1
        local_C_row[j] = 0;
    }

	      {

    hls::stream<int>      s_k("s_k");
    hls::stream<data_t>   s_valA("s_valA");
    hls::stream<int>      s_j("s_j");
    hls::stream<data_t>   s_valB("s_valB");
    hls::stream<data_t>   s_a_val("s_a_val");
    #pragma HLS STREAM variable=s_k    depth=59
    #pragma HLS STREAM variable=s_valA depth=59
    #pragma HLS STREAM variable=s_j    depth=3059
    #pragma HLS STREAM variable=s_valB depth=3059
    #pragma HLS STREAM variable=s_a_val depth=3059


    #pragma HLS DATAFLOW
    stage1_function(i, row_ptr_A_local, column_indices_A_local, values_A_local, s_k, s_valA);
    stage2_function(i, row_ptr_A_local, column_indices_A_local, values_A_local,
                    col_ptr_B_local, row_indices_B_local, values_B_local,
                    s_k, s_valA, s_j, s_valB, s_a_val);
    stage3_function(i, row_ptr_A_local, column_indices_A_local, values_A_local,
                    col_ptr_B_local, row_indices_B_local, values_B_local,
                    s_j, s_valB, s_a_val, local_C_row);
	      }
}


void sparse_matrix_multiply_HLS(
    data_t values_A[N * M],
    int    column_indices_A[N * M],
    int    row_ptr_A[N + 1],
    data_t values_B[M * K],
    int    row_indices_B[M * K],
    int    col_ptr_B[M + 1],
    data_t C[N][K]
) {
#pragma HLS interface m_axi port=values_A         offset=slave bundle=mem1
#pragma HLS interface m_axi port=column_indices_A   offset=slave bundle=mem1
#pragma HLS interface m_axi port=row_ptr_A          offset=slave bundle=mem1

#pragma HLS interface m_axi port=values_B         offset=slave bundle=mem2
#pragma HLS interface m_axi port=row_indices_B    offset=slave bundle=mem2
#pragma HLS interface m_axi port=col_ptr_B        offset=slave bundle=mem2

#pragma HLS interface m_axi port=C                  offset=slave bundle=mem3
#pragma HLS interface s_axilite port=return


    data_t values_A_local[N * M];
    int    column_indices_A_local[N * M];
    int    row_ptr_A_local[N + 1];

    data_t values_B_local[M * K];
    int    row_indices_B_local[M * K];
    int    col_ptr_B_local[M + 1];


    data_t local_C[N][K];



    {
#pragma HLS DATAFLOW
    copy_A_to_local(
        values_A, column_indices_A, row_ptr_A,
        values_A_local, column_indices_A_local, row_ptr_A_local
    );
    copy_B_to_local(
        values_B, row_indices_B, col_ptr_B,
        values_B_local, row_indices_B_local, col_ptr_B_local
    );
    }


    process_rows: for (int i = 0; i < N; i++) {
    #pragma HLS UNROLL factor=4  
        process_row(i, row_ptr_A_local, column_indices_A_local, values_A_local,
                    col_ptr_B_local, row_indices_B_local, values_B_local, local_C[i]);
    }


    copy_C: for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
        #pragma HLS PIPELINE II=1
            C[i][j] = local_C[i][j];
        }
    }
}

