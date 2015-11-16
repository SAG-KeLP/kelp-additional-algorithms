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
public class L2R_L2_SvcFunction implements TronFunction {

	protected final Problem prob;
	protected final double[] C;
	protected final int[] I;
	protected final double[] z;

	protected int sizeI;

	public L2R_L2_SvcFunction(Problem prob, double[] C) {
		int l = prob.l;

		this.prob = prob;

		z = new double[l];
		I = new int[l];
		this.C = C;
	}

	public double fun(double[] w) {
		int i;
		double f = 0;
		double[] y = prob.y;
		int l = prob.l;
		int w_size = get_nr_variable();

		Xv(w, z);

		for (i = 0; i < w_size; i++)
			f += w[i] * w[i];
		f /= 2.0;
		for (i = 0; i < l; i++) {
			z[i] = y[i] * z[i];
			double d = 1 - z[i];
			if (d > 0)
				f += C[i] * d * d;
		}

		return (f);
	}

	public int get_nr_variable() {
		return prob.n;
	}

	public void grad(double[] w, double[] g) {
		double[] y = prob.y;
		int l = prob.l;
		int w_size = get_nr_variable();

		sizeI = 0;
		for (int i = 0; i < l; i++) {
			if (z[i] < 1) {
				z[sizeI] = C[i] * y[i] * (z[i] - 1);
				I[sizeI] = i;
				sizeI++;
			}
		}
		subXTv(z, g);

		for (int i = 0; i < w_size; i++)
			g[i] = w[i] + 2 * g[i];
	}

	public void Hv(double[] s, double[] Hs) {
		int i;
		int w_size = get_nr_variable();
		double[] wa = new double[sizeI];

		subXv(s, wa);
		for (i = 0; i < sizeI; i++)
			wa[i] = C[I[i]] * wa[i];

		subXTv(wa, Hs);
		for (i = 0; i < w_size; i++)
			Hs[i] = s[i] + 2 * Hs[i];
	}

	protected void subXTv(double[] v, double[] XTv) {
		int i;
		int w_size = get_nr_variable();

		for (i = 0; i < w_size; i++)
			XTv[i] = 0;

		for (i = 0; i < sizeI; i++) {
			for (LibLinearFeature s : prob.x[I[i]]) {
				XTv[s.getIndex() - 1] += v[i] * s.getValue();
			}
		}
	}

	private void subXv(double[] v, double[] Xv) {

		for (int i = 0; i < sizeI; i++) {
			Xv[i] = 0;
			for (LibLinearFeature s : prob.x[I[i]]) {
				Xv[i] += v[s.getIndex() - 1] * s.getValue();
			}
		}
	}

	protected void Xv(double[] v, double[] Xv) {

		for (int i = 0; i < prob.l; i++) {
			Xv[i] = 0;
			for (LibLinearFeature s : prob.x[i]) {
				Xv[i] += v[s.getIndex() - 1] * s.getValue();
			}
		}
	}

}
