__kernel void picalc(__global float* a, __global float* b, __global float* c, int n)
        {
        unsigned int L= get_local_size(0);
        unsigned int gid = get_global_id(0);

        for(int i=gid;i<n;i=i+L){
        if (a[i]*a[i]+ b[i]*b[i] < 100*100)
           {
            c[i] = 1;
           }
        else
           {
            c[i]=0;
           }
        }
        }



