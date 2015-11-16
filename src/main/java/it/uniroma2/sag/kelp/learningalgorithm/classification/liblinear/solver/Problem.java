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
import gnu.trove.map.hash.TIntObjectHashMap;
import gnu.trove.map.hash.TObjectIntHashMap;
import it.uniroma2.sag.kelp.data.dataset.Dataset;
import it.uniroma2.sag.kelp.data.example.Example;
import it.uniroma2.sag.kelp.data.example.SimpleExample;
import it.uniroma2.sag.kelp.data.label.Label;
import it.uniroma2.sag.kelp.data.representation.Representation;
import it.uniroma2.sag.kelp.data.representation.Vector;
import it.uniroma2.sag.kelp.data.representation.vector.DenseVector;
import it.uniroma2.sag.kelp.data.representation.vector.SparseVector;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Map;

/**
 * <p>
 * Describes the problem
 * </p>
 * 
 * For example, if we have the following training data:
 * 
 * <pre>
 *  LABEL       ATTR1   ATTR2   ATTR3   ATTR4   ATTR5
 *  -----       -----   -----   -----   -----   -----
 *  1           0       0.1     0.2     0       0
 *  2           0       0.1     0.3    -1.2     0
 *  1           0.4     0       0       0       0
 *  2           0       0.1     0       1.4     0.5
 *  3          -0.1    -0.2     0.1     1.1     0.1
 * 
 *  and bias = 1, then the components of problem are:
 * 
 *  l = 5
 *  n = 6
 * 
 *  y -&gt; 1 2 1 2 3
 * 
 *  x -&gt; [ ] -&gt; (2,0.1) (3,0.2) (6,1) (-1,?)
 *       [ ] -&gt; (2,0.1) (3,0.3) (4,-1.2) (6,1) (-1,?)
 *       [ ] -&gt; (1,0.4) (6,1) (-1,?)
 *       [ ] -&gt; (2,0.1) (4,1.4) (5,0.5) (6,1) (-1,?)
 *       [ ] -&gt; (1,-0.1) (2,-0.2) (3,0.1) (4,1.1) (5,0.1) (6,1) (-1,?)
 * </pre>
 */
public class Problem {

	public enum LibLinearSolverType {
		CLASSIFICATION, REGRESSION
	}

	public TObjectIntHashMap<Object> featureDict = new TObjectIntHashMap<Object>();

	public TIntObjectHashMap<Object> featureInverseDict = new TIntObjectHashMap<Object>();

	/** the number of training data */
	public int l;

	/** the number of features (including the bias feature if bias &gt;= 0) */
	public int n;

	/** an array containing the target values */
	public double[] y;
	/** array of sparse feature nodes */
	public LibLinearFeature[][] x;

	/**
	 * If bias &gt;= 0, we assume that one additional feature is added to the
	 * end of each data instance
	 */
	public double bias;

	private boolean isInputDense;

	public Problem(Dataset dataset, String reprentationName, Label label,
			LibLinearSolverType solverType) {

		this.l = dataset.getNumberOfExamples();
		this.y = new double[l];
		this.x = new LibLinearFeature[l][];

		ArrayList<Vector> vectorlist = new ArrayList<Vector>();

		if (dataset.getExamples().get(0).getRepresentation(reprentationName) instanceof DenseVector)
			isInputDense = true;

		int i = 0;
		for (Example e : dataset.getExamples()) {
			SimpleExample simpleExample = (SimpleExample) e;
			Representation r = simpleExample
					.getRepresentation(reprentationName);
			Vector vector = (Vector) r;

			vectorlist.add(vector);

			if (solverType == LibLinearSolverType.CLASSIFICATION) {
				if (e.isExampleOf(label))
					y[i] = 1;
				else
					y[i] = -1;
			} else {
				y[i] = e.getRegressionValue(label);
			}

			i++;
		}

		initializeExamples(vectorlist);

	}

	private DenseVector getDenseW(double[] w) {
		double[] tmp = new double[w.length - 1];
		for (int i = 0; i < w.length - 1; i++) {
			tmp[i] = w[i];
		}
		return new DenseVector(tmp);
	}

	private SparseVector getSparseW(double[] w) {
		SparseVector res = new SparseVector();

		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < w.length - 1; i++) {
			sb.append(this.featureInverseDict.get(i + 1) + ":" + w[i] + " ");
		}
		sb.append("__LIB_LINEAR_BIAS_:" + w[w.length - 1]);

		try {
			res.setDataFromText(sb.toString().trim());
		} catch (IOException e) {
			e.printStackTrace();
			return null;
		}
		return res;
	}

	public Vector getW(double[] w) {
		if (isInputDense) {
			return getDenseW(w);
		}
		return getSparseW(w);
	}

	public void initializeExamples(ArrayList<Vector> vectorlist) {
		if (isInputDense) {
			initializeExamplesDense(vectorlist);
		} else {
			initializeExamplesSparse(vectorlist);
		}
	}

	private void initializeExamplesDense(ArrayList<Vector> vectorlist) {
		for (int vectorId = 0; vectorId < vectorlist.size(); vectorId++) {
			DenseVector denseVector = (DenseVector) (vectorlist.get(vectorId));
			if (vectorId == 0) {
				bias = 0;
				n = denseVector.getNumberOfFeatures() + 1;
			}
			this.x[vectorId] = new LibLinearFeatureNode[denseVector
					.getNumberOfFeatures()];
			for (int j = 0; j < denseVector.getNumberOfFeatures(); j++)
				this.x[vectorId][j] = new LibLinearFeatureNode(j + 1,
						denseVector.getFeatureValue(j));
		}
	}

	private void initializeExamplesSparse(ArrayList<Vector> vectorlist) {
		/*
		 * Building dictionary
		 */
		int featureIndex = 1;
		for (Vector v : vectorlist) {
			//for (String dimLabel : v.getActiveFeatures().keySet()) {
			for (Object dimLabel : v.getActiveFeatures().keySet()) {
				if (!featureDict.containsKey(dimLabel)) {
					featureDict.put(dimLabel, featureIndex);
					featureInverseDict.put(featureIndex, dimLabel);
					featureIndex++;
					// System.out.println(featureIndex + " " + dimLabel);
				}
			}
		}

		/*
		 * Initialize the object
		 */
		n = featureDict.size() + 1;
		bias = 0;
		int i = 0;
		for (Vector v : vectorlist) {
			Map<?, Number> activeFeatures = v.getActiveFeatures();
			this.x[i] = new LibLinearFeatureNode[activeFeatures.size()];
			int j = 0;
			for (Object dimLabel : activeFeatures.keySet()) {
				this.x[i][j] = new LibLinearFeatureNode(
						featureDict.get(dimLabel), activeFeatures.get(dimLabel).doubleValue());
				j++;
			}
			i++;
		}
	}

}
