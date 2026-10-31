// Verification: empty normalization graphs must not invoke chip work.
// usage: empty-graph LIB [DEVICE], default DEVICE OmegaGB10.
#include "ggml.h"
#include "ggml-backend.h"
#include <stdio.h>
int main(int argc,char **argv) {
 if(argc<2 || argc>3)return 2;
 if(!ggml_backend_load(argv[1]))return 2;
 ggml_backend_t b=ggml_backend_init_by_name(argc==3 ? argv[2] : "OmegaGB10",NULL);
 if(!b)return 2;
 struct ggml_init_params p={.mem_size=1024*1024,.mem_buffer=NULL,.no_alloc=false};
 struct ggml_context *c=ggml_init(p);
 struct ggml_tensor *x=ggml_new_tensor_2d(c,GGML_TYPE_F32,128,0);
 struct ggml_tensor *y=ggml_rms_norm(c,x,0.00001f);
 ggml_set_output(y);
 struct ggml_cgraph *g=ggml_new_graph(c);
 ggml_build_forward_expand(g,y);
 fprintf(stderr,"Empty RMS: elements=%lld rows=%lld dim=%lld\n",(long long)ggml_nelements(y),(long long)ggml_nrows(y),(long long)y->ne[0]);
 int r=ggml_backend_graph_compute(b,g);
 ggml_backend_free(b);ggml_free(c);
 return r!=GGML_STATUS_SUCCESS;
}
