// This is a C implementation of the trueanomaly function to increase speed.

#include <math.h>
#include <stdlib.h>
#include "trueanomaly.h"
#include <stdio.h>

int trueanomaly(float *M, int n, float ecc, float *nu, int niterationmax, float tol)
{
    // Set upper limit for eccentricity
    if (ecc > 0.99)
        ecc = 0.99;

    // Newton-Raphson method
    for (int i = 0; i < n; i++)
    {
        float E = M[i];
        float E0 = M[i];
        int niteration = 0;

        while (fabs(E - E0) > tol || niteration == 0)
        {
            E0 = E;

            float ff = E - ecc * sin(E) - M[i];
            float dff = 1 - ecc * cos(E);

            // Use Newton method
            E = E0 - ff / dff;

            niteration += 1;
            if (niteration >= niterationmax)
                return -1;
        }
        // Compute true anomaly
        nu[i] = 2. * atan2(sqrt(1. + ecc) * sin(E / 2.), sqrt(1. - ecc) * cos(E / 2.));
    }

    return 0;
}