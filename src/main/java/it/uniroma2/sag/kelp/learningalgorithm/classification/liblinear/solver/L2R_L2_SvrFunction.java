/*
 * Copyright 2014 Simone Filice and Giuseppe Castellucci and Danilo Croce and Roberto Basili
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package it.uniroma2.sag.kelp.learningalgorithm.classification.liblinear.solver;

/**
 * NOTE: This code has been adapted from the Java port of the original LIBLINEAR
 * C++ sources. Original Java sources (v 1.94) are available at
 * <code>http://liblinear.bwaldvogel.de</code>
 * 
 * @author Danilo Croce
 */
public class L2R_L2_SvrFunction extends L2R_L2_SvcFunction {

    private double p;

    public L2R_L2_SvrFunction( Problem prob, double[] C, double p ) {
        super(prob, C);
        this.p = p;
    }

    @Override
    public double fun(double[] w) {
        double f = 0;
        double[] y = prob.y;
        int l = prob.l;
        int w_size = get_nr_variable();
        double d;

        Xv(w, z);

        for (int i = 0; i < w_size; i++)
            f += w[i] * w[i];
        f /= 2;
        for (int i = 0; i < l; i++) {
            d = z[i] - y[i];
            if (d < -p)
                f += C[i] * (d + p) * (d + p);
            else if (d > p) f += C[i] * (d - p) * (d - p);
        }

        return f;
    }

    @Override
    public void grad(double[] w, double[] g) {
        double[] y = prob.y;
        int l = prob.l;
        int w_size = get_nr_variable();

        sizeI = 0;
        for (int i = 0; i < l; i++) {
            double d = z[i] - y[i];

            // generate index set I
            if (d < -p) {
                z[sizeI] = C[i] * (d + p);
                I[sizeI] = i;
                sizeI++;
            } else if (d > p) {
                z[sizeI] = C[i] * (d - p);
                I[sizeI] = i;
                sizeI++;
            }

        }
        subXTv(z, g);

        for (int i = 0; i < w_size; i++)
            g[i] = w[i] + 2 * g[i];

    }

}