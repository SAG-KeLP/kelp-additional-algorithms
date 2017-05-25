/*
 * Copyright 2016 Simone Filice and Giuseppe Castellucci and Danilo Croce and Roberto Basili
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

package it.uniroma2.sag.kelp.linearization.nystrom;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import org.ejml.data.DenseMatrix64F;
import org.ejml.factory.DecompositionFactory;
import org.ejml.factory.SingularValueDecomposition;
import org.ejml.ops.CommonOps;
import org.ejml.ops.SingularOps;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonTypeName;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;

import it.uniroma2.sag.kelp.data.dataset.Dataset;
import it.uniroma2.sag.kelp.data.dataset.SimpleDataset;
import it.uniroma2.sag.kelp.data.example.Example;
import it.uniroma2.sag.kelp.data.example.SimpleExample;
import it.uniroma2.sag.kelp.data.representation.Representation;
import it.uniroma2.sag.kelp.data.representation.vector.DenseVector;
import it.uniroma2.sag.kelp.kernel.Kernel;
import it.uniroma2.sag.kelp.linearization.LinearizationFunction;
import it.uniroma2.sag.kelp.utils.FileUtils;

/**
 * This class implements the Nystrom Method to approximate the implicit space
 * underlying a Kernel Function, thus producing a low-dimensional dense
 * representation. <br>
 * As an example, given a <code>Dataset</code> of examples represented through
 * tree structures and a tree kernel function, this class allows deriving a
 * linearized dataset at a given dimensionality. <br>
 * <br>
 * If you use this class, <b>please cite</b>: <br>
 * <li>Danilo Croce and Roberto Basili. Large-scale Kernel-based Language
 * Learning through the Ensemble Nystrom methods. In Proceedings of ECIR 2016.
 * Padova, Italy, 2016 <br>
 * 
 * @author Danilo Croce
 */
@JsonTypeName("nystrom")
public class NystromMethod implements LinearizationFunction {

	private Logger logger = LoggerFactory.getLogger(NystromMethod.class);

	@JsonIgnore
	public static final float ESPILON = 0.00001f;

	/**
	 * Load a Nystrom-based projection function from a file
	 * 
	 * @param inputFilePath
	 *            the input file
	 * @return Nystrom-based projection function
	 * @throws FileNotFoundException
	 * @throws IOException
	 */
	public static NystromMethod load(String inputFilePath) throws FileNotFoundException, IOException {
		ObjectMapper mapper = new ObjectMapper();
		InputStreamReader inS = new InputStreamReader(FileUtils.createInputStream(inputFilePath), "utf8");
		NystromMethod readValue = mapper.readValue(inS, NystromMethod.class);
		inS.close();
		return readValue;
	}

	/**
	 * The kernel function over the implicit space linearized by the Nystrom
	 * Method
	 */
	private Kernel kernel;

	/**
	 * The examples used as landmarks
	 */
	private List<Example> landmarks;

	/**
	 * The linear projection matrix used for the serialization. It is the
	 * Moore-Penrose inverse of the subset of kernel matrix estimated among
	 * landmarks
	 */
	private List<Double> projectionMatrix;

	@JsonIgnore
	private boolean debug = false;

	@JsonIgnore
	private DenseMatrix64F USigmaSquare;

	@JsonIgnore
	private DenseMatrix64F kernelValuesToProject;

	/**
	 * The expected rank of the space representing the linearized examples
	 */
	private int rank;

	/**
	 * 
	 */
	public NystromMethod() {

	}

	/**
	 * Constructor of NystromMethod. The expected rank is automatically set to
	 * the size of <code>landmarks</code>
	 * 
	 * @param landmarks
	 *            The set of examples used as landmarks
	 * @param kernel
	 *            The kernel function
	 * @throws InstantiationException
	 */
	public NystromMethod(List<Example> landmarks, Kernel kernel) throws InstantiationException {
		this(landmarks, kernel, landmarks.size());
	}

	/**
	 * @param landmarks
	 *            The set of examples used as landmarks
	 * @param kernel
	 *            The kernel function
	 * @param expectedRank
	 *            The expected rank of the space representing the linearized
	 *            examples
	 * @throws InstantiationException
	 */
	public NystromMethod(List<Example> landmarks, Kernel kernel, int expectedRank) throws InstantiationException {

		this.kernel = kernel;

		this.landmarks = landmarks;

		int m = this.landmarks.size();

		this.rank = expectedRank;

		if (expectedRank > m) {
			debug("Expected Rank (" + expectedRank + ") and it is higher than m (" + m + "). It will be reduced to m.");
			this.rank = m;
		}

		calculateProjMatrix();
	}

	/**
	 * Estimates the projection matrix
	 */
	private void calculateProjMatrix() {
		int m = landmarks.size();

		DenseMatrix64F W = new DenseMatrix64F(m, m);

		info("Numbero of landmarks:\t" + m);

		info("Building W...");
		for (int i = 0; i < m; i++) {
			if ((i + 1) % 100 == 0)
				info("Evaluated " + (i + 1) + " landmarks.");
			for (int j = i; j < m; j++) {
				float k = this.kernel.innerProduct(landmarks.get(i), landmarks.get(j));
				W.set(i, j, k);
				W.set(j, i, k);
				if (i == j)
					W.set(i, i, W.get(i, i));
			}
		}
		debug("W\n" + W);

		info("SVD Decomposition...");
		SingularValueDecomposition<DenseMatrix64F> svd = DecompositionFactory.svd(W.getNumRows(), W.getNumRows(), true,
				true, false);

		info("Decompostion completed");

		if (!svd.decompose(W))
			throw new RuntimeException("Decomposition failed");

		DenseMatrix64F U = svd.getU(null, false);
		DenseMatrix64F V = svd.getV(null, false);

		debug("U\n" + U);
		DenseMatrix64F SigmaSquare = svd.getW(null);
		debug("Sigma^(1/2)\n" + SigmaSquare);

		SingularOps.descendingOrder(U, false, SigmaSquare, V, false);

		for (int i = 0; i < m; i++) {
			debug("Sigma\t" + i + "\t" + SigmaSquare.get(i, i));
		}

		for (int i = 0; i < this.rank; i++) {
			if (SigmaSquare.get(i, i) / SigmaSquare.get(0, 0) < ESPILON) {
				this.rank = i;
				break;
			}
		}
		info("Final matrix rank:\t" + rank);

		for (int i = rank; i < m; i++) {
			SigmaSquare.set(i, i, 0);
		}

		info("Calculating Projection matrix...");
		for (int i = 0; i < m; i++) {
			double sigma = SigmaSquare.get(i, i);
			if (sigma > 0)
				sigma = 1 / Math.sqrt(sigma);
			SigmaSquare.set(i, i, sigma);
		}

		this.USigmaSquare = new DenseMatrix64F(m, m);
		CommonOps.mult(U, SigmaSquare, USigmaSquare);
		debug("U*Sigma^(1/2)\n" + USigmaSquare);

		this.projectionMatrix = new ArrayList<Double>();

		for (int i = 0; i < m; i++) {
			for (int j = 0; j < rank; j++) {
				this.projectionMatrix.add(this.USigmaSquare.get(i, j));
			}
		}
		USigmaSquare = null;

	}

	/**
	 * It derives an array of doubles containing the linearized representation
	 * 
	 * @param example
	 *            the input example
	 * @return the linearized representaton as array of double
	 */
	protected double[] calculateVector(Example example) {
		DenseMatrix64F projectedVector = getDenseMatrix64F(example);

		return projectedVector.data;
	}

	private DenseMatrix64F getDenseMatrix64F(Example example) {
		
		int m = landmarks.size();
		
		if (USigmaSquare == null) {
			USigmaSquare = new DenseMatrix64F(m, m);
			double[] data = new double[m * m];
			int origId=0;
			int arrayId=0;
			for (int r = 0; r < m; r++) {
				for(int j=0; j<rank; j++)
					data[arrayId++] = projectionMatrix.get(origId++);
				for(int j=rank; j<m; j++)
					data[arrayId++] = 0;
			}
			USigmaSquare.setData(data);
		}

		if (kernelValuesToProject == null) {
			kernelValuesToProject = new DenseMatrix64F(1, m);
		}

		DenseMatrix64F projectedVector = new DenseMatrix64F(1, m);

		for (int j = 0; j < m; j++) {
			float k = this.kernel.innerProduct(example, landmarks.get(j));
			kernelValuesToProject.set(0, j, k);
		}

		CommonOps.mult(kernelValuesToProject, USigmaSquare, projectedVector);
		return projectedVector;
	}

	private void debug(String string) {
		logger.debug(string);
	}

	/**
	 * @return The kernel function over the implicit space linearized by the
	 *         Nystrom Method
	 */
	public Kernel getKernel() {
		return kernel;
	}

	/**
	 * @return The examples used as landmarks
	 */
	public List<Example> getLandmarks() {
		return landmarks;
	}

	@Override
	public SimpleDataset getLinearizedDataset(Dataset dataset, String representationName) {

		SimpleDataset resDataset = new SimpleDataset();

		int count = 1;
		for (Example e : dataset.getExamples()) {
			resDataset.addExample(getLinearizedExample(e, representationName));
			if (count % 1000 == 0) {
				info("Projected " + count + " examples.");
			}
			count++;
		}

		return resDataset;
	}

	@Override
	public Example getLinearizedExample(Example example, String representationName) {
		DenseMatrix64F denseMatrix64F = getDenseMatrix64F(example);
		DenseVector denseVector = new DenseVector(denseMatrix64F);

		HashMap<String, Representation> representations = new HashMap<String, Representation>();
		representations.put(representationName, denseVector);

		Example res = new SimpleExample(example.getLabels(), representations);
		res.setRegressionValues(example.getRegressionValues());
		return res;
	}

	@Override
	public DenseVector getLinearRepresentation(Example example) {
		double[] projectedVector = calculateVector(example);
		return new DenseVector(projectedVector);
	}

	/**
	 * @return The linear projection matrix used for the serialization. It is
	 *         the Moore-Penrose inverse of the subset of kernel matrix
	 *         estimated among landmarks
	 */
	public List<Double> getProjectionMatrix() {
		return projectionMatrix;
	}

	/**
	 * @return The expected rank of the space representing the linearized
	 *         examples
	 */
	public int getRank() {
		return rank;
	}

	private void info(String string) {
		logger.info(string);
	}

	/**
	 * Save a Nystrom-based projection function in a file. If the .gz suffix is
	 * used a compressed file is obtained
	 * 
	 * @param outputFilePath
	 *            the output file path
	 * @throws FileNotFoundException
	 * @throws IOException
	 */
	public void save(String outputFilePath) throws FileNotFoundException, IOException {
		ObjectMapper mapper = new ObjectMapper();
		mapper.enable(SerializationFeature.INDENT_OUTPUT);

		OutputStreamWriter out = new OutputStreamWriter(FileUtils.createOutputStream(outputFilePath), "utf8");
		mapper.writeValue(out, this);
		out.close();
	}

	/**
	 * @param kernel
	 *            The kernel function
	 */
	public void setKernel(Kernel kernel) {
		this.kernel = kernel;
	}

	/**
	 * @param landmarks
	 *            The landmarks
	 */
	public void setLandmarks(List<Example> landmarks) {
		this.landmarks = landmarks;
	}

	/**
	 * @param projectionMatrix
	 *            The projection matrix
	 */
	public void setProjectionMatrix(List<Double> projectionMatrix) {
		this.projectionMatrix = projectionMatrix;
	}

	/**
	 * @param rank
	 *            The expected rank
	 */
	public void setRank(int rank) {
		this.rank = rank;
	}

	@Override
	public int getEmbeddingSize() {
		return landmarks.size();
	}

}
