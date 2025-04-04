#include <hls_math.h>

#include <hls_stream.h>

#include <cstring>

#include "dcl.h"
 


#define B 4

#define N 100

#define dk 128

#define dv 128

#define CHUNK_SIZE 1
 

typedef struct {

    fixed_t data[N];

} row_t;
 


void compute_qk(

    fixed_t Q_block[B][CHUNK_SIZE][dk],

    const fixed_t K_local[B][N][dk],

    hls::stream<row_t> &qk_stream)

{

    for (int b = 0; b < B; b++) {


        row_t row;


        const ap_fixed<32,8> scale = 0.0884; 

        for (int j = 0; j < N; j++) {

#pragma HLS PIPELINE II=1

          ap_fixed<32,8> sum = 0;

          for (int k = 0; k < dk; k++) {

#pragma HLS UNROLL factor=16

            sum += Q_block[b][0][k] * K_local[b][j][k];

          }

          row.data[j] = sum * scale;

        }

        qk_stream.write(row);

      }

    }

 

void softmax_stage(

    hls::stream<row_t> &qk_stream,

    hls::stream<row_t> &softmax_stream)

{

    for (int b = 0; b < B; b++) {


        row_t row = qk_stream.read();


        ap_fixed<32,8> max_val = row.data[0];

        for (int j = 1; j < N; j++) {

#pragma HLS PIPELINE II=1

          if (row.data[j] > max_val)

            max_val = row.data[j];

        }


        ap_fixed<32,8> sum = 0;

        for (int j = 0; j < N; j++) {

#pragma HLS PIPELINE II=1

          row.data[j] = hls::exp(row.data[j] - max_val);

          sum += row.data[j];

        }


        for (int j = 0; j < N; j++) {

#pragma HLS PIPELINE II=1

          row.data[j] = row.data[j] / sum;

        }

        softmax_stream.write(row);

      }

    }

 

void attention_v_stage(

    hls::stream<row_t> &softmax_stream,

    const fixed_t V_local[B][N][dv],

    fixed_t Output_blk[B][CHUNK_SIZE][dv])

{

    for (int b = 0; b < B; b++) {


        row_t attention_row = softmax_stream.read();

        for (int j = 0; j < dv; j++) {

#pragma HLS PIPELINE II=1

          ap_fixed<32,8> sum = 0;

          for (int k = 0; k < N; k++) {

#pragma HLS UNROLL factor=10

            sum += attention_row.data[k] * V_local[b][k][j];

          }

          Output_blk[b][0][j] = sum;

        }

      }

    }

 

void compute_dflow(

    int block,

    fixed_t Q_block[B][CHUNK_SIZE][dk],

    fixed_t K_local[B][N][dk],

    fixed_t V_local[B][N][dv],

    fixed_t Output_blk[B][CHUNK_SIZE][dv])

{

    hls::stream<row_t> qk_stream("qk_stream");

    hls::stream<row_t> softmax_stream("softmax_stream");

#pragma HLS stream variable=qk_stream depth=4

#pragma HLS stream variable=softmax_stream depth=4
 
#pragma HLS DATAFLOW
 
    compute_qk(Q_block, K_local, qk_stream);

    softmax_stage(qk_stream, softmax_stream);

    attention_v_stage(softmax_stream, V_local, Output_blk);

}
 

void compute_attention_HLS(

     fixed_t Q[B][N][dk],

     fixed_t K_in[B][N][dk],

     fixed_t V_in[B][N][dv],

     fixed_t Output[B][N][dv])

{

#pragma HLS INTERFACE m_axi port=Q offset=slave bundle=gmem0

#pragma HLS INTERFACE m_axi port=K_in offset=slave bundle=gmem1

#pragma HLS INTERFACE m_axi port=V_in offset=slave bundle=gmem2

#pragma HLS INTERFACE m_axi port=Output offset=slave bundle=gmem3

#pragma HLS INTERFACE s_axilite port=return
 

    fixed_t K_local[B][N][dk];

    fixed_t V_local[B][N][dv];
 

    for (int b = 0; b < B; b++) {

      for (int i = 0; i < N; i++) {

        for (int k = 0; k < dk; k++) {

#pragma HLS PIPELINE II=1

          K_local[b][i][k] = K_in[b][i][k];

        }

      }

    }
 

    for (int b = 0; b < B; b++) {

      for (int i = 0; i < N; i++) {

        for (int j = 0; j < dv; j++) {

#pragma HLS PIPELINE II=1

          V_local[b][i][j] = V_in[b][i][j];

        }

      }

    }
 

    fixed_t Q_block[B][CHUNK_SIZE][dk];

    fixed_t Output_blk[B][CHUNK_SIZE][dv];
 

    for (int block = 0; block < N; block += CHUNK_SIZE) {


      for (int b = 0; b < B; b++) {


#pragma HLS PIPELINE II=1

          for (int k = 0; k < dk; k++) {

            Q_block[b][0][k] = Q[b][block + 0][k];

          }

        }


      compute_dflow(block, Q_block, K_local, V_local, Output_blk);
 

      for (int b = 0; b < B; b++) {


#pragma HLS PIPELINE II=1

          for (int j = 0; j < dv; j++) {

            Output[b][block + 0][j] = Output_blk[b][0][j];

          }

        }

      }

    }


 
