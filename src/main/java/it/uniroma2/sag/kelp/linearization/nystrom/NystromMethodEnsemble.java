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
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;

import it.uniroma2.sag.kelp.data.dataset.Dataset;
import it.uniroma2.sag.kelp.data.dataset.SimpleDataset;
import it.uniroma2.sag.kelp.data.example.Example;
import it.uniroma2.sag.kelp.data.example.SimpleExample;
import it.uniroma2.sag.kelp.data.representation.Representation;
import it.uniroma2.sag.kelp.data.representation.Vector;
import it.uniroma2.sag.kelp.data.representation.vector.DenseVector;
import it.uniroma2.sag.kelp.kernel.Kernel;
import it.uniroma2.sag.kelp.linearization.LinearizationFunction;
import it.uniroma2.sag.kelp.utils.FileUtils;

/**
 * This class implements the Ensemble Nystrom Method to approximate the implicit
 * space underlying a Kernel Function, thus producing a low-dimensional dense
 * representation. <br>
 * Several projection functions can be defined according to the Nystrom Method
 * and they can be use together to improve approximation quality. <br>
 *
 * More details can be found in the following paper. If you use this class,
 * <b>please cite</b>: <br>
 * <li>Danilo Croce and Roberto Basili. Large-scale Kernel-based Language
 * Learning through the Ensemble Nystrom methods. In Proceedings of ECIR 2016.
 * Padova, Italy, 2016 <br>
 * 
 * @author Danilo Croce
 */
public class NystromMethodEnsemble extends ArrayList<NystromMethod> implements LinearizationFunction {

	private Logger logger = LoggerFactory.getLogger(NystromMethodEnsemble.class);

	/**
	 * @param listOflandmarks
	 *            The set of landmarks to build the projection functions based
	 *            on the Nystrom Method
	 * @param kernel
	 *            The kernel function
	 * @throws InstantiationException
	 */
	public NystromMethodEnsemble(List<List<Example>> listOflandmarks, Kernel kernel) throws InstantiationException {
		for (List<Example> landmarks : listOflandmarks) {
			this.add(new NystromMethod(landmarks, kernel));
		}
	}

	public NystromMethodEnsemble() {

	}

	/**
	 * 
	 */
	private static final long serialVersionUID = -7379502720881996512L;

	/**
	 * Load an Ensemble of Nystrom projectors saved on file.
	 * 
	 * @param inputFilePath
	 *            The input file path
	 * @return
	 * @throws FileNotFoundException
	 * @throws IOException
	 */
	public static NystromMethodEnsemble load(String inputFilePath) throws FileNotFoundException, IOException {
		ObjectMapper mapper = new ObjectMapper();
		InputStream inputStream = FileUtils.createInputStream(inputFilePath);
		return mapper.readValue(inputStream, NystromMethodEnsemble.class);
	}

	/**
	 * Save an Ensemble of Nystrom projectors on file.
	 * 
	 * @param outputFilePath
	 *            The output file name
	 * @throws FileNotFoundException
	 * @throws IOException
	 */
	public void save(String outputFilePath) throws FileNotFoundException, IOException {
		ObjectMapper mapper = new ObjectMapper();
		mapper.enable(SerializationFeature.INDENT_OUTPUT);

		OutputStream outputStream = FileUtils.createOutputStream(outputFilePath);
		mapper.writeValue(outputStream, this);
	}

	/**
	 * Given an example, this method produces a new <code>Example
	 * <code> containing a single representation, i.e. a dense vector that is
	 * the concatenation of the vectors obtained by each projection
	 * function used in the Ensemble. The <code>label</code>s are copied from
	 * the input example.
	 * 
	 * @param example
	 *            the input example
	 * @param newRepresentationName
	 *            the identifier of the new dense vector
	 * @return a new <code>Example <code> containing a single representation,
	 *         i.e. a dense vector that is the concatenation of the vectors
	 *         obtained by each projection function used in the Ensemble
	 * @throws InstantiationException
	 */
	public Example linearizeByEnsembleAndJuxtaposition(Example example, String newRepresentationName)
			throws InstantiationException {
		if (size() == 1) {
			return get(0).getLinearizedExample(example, newRepresentationName);
		}

		ArrayList<Float> weights = new ArrayList<Float>();
		for (int i = 0; i < size(); i++) {
			weights.add(1f);
		}
		DenseVector denseVector = getDenseVectorByEnsembleAndJuxtaposition(example, weights);

		HashMap<String, Representation> representations = new HashMap<String, Representation>();
		representations.put(newRepresentationName, denseVector);

		return new SimpleExample(example.getLabels(), representations);
	}

	/**
	 * Given an example, this method produces a <code>DenseVector</code> that is
	 * the concatenation of the vectors obtained by each projection functions
	 * used in the Ensemble.
	 * 
	 * @param example
	 *            the input example
	 * @return the concatenation of the vectors obtained by each projection
	 *         functions used in the Ensemble.
	 * @throws InstantiationException
	 */
	public DenseVector getDenseVectorByEnsembleAndJuxtaposition(Example example) throws InstantiationException {

		List<Float> weights = new ArrayList<Float>();

		float weight = 1f / (float) size();
		for (int i = 0; i < size(); i++) {
			weights.add(weight);
		}

		return getDenseVectorByEnsembleAndJuxtaposition(example, weights);
	}

	/**
	 * Given an example, this method produces a <code>DenseVector</code> that is
	 * the concatenation of the vectors obtained by each projection functions
	 * used in the Ensemble. Each vector used in the concatenation is multiplied
	 * by a corresponding weight
	 * 
	 * @param example
	 *            The input example
	 * @param weights
	 *            the weights applied to each vector before the concatenation
	 * @return
	 * @throws InstantiationException
	 */
	public DenseVector getDenseVectorByEnsembleAndJuxtaposition(Example example, List<Float> weights)
			throws InstantiationException {

		int m = size();
		int newDimensionality = 0;
		double[][] ensembleBuffer = new double[m][];
		for (int i = 0; i < size(); i++) {
			ensembleBuffer[i] = get(i).calculateVector(example);
			newDimensionality += ensembleBuffer[i].length;
		}

		double[] newVector = new double[newDimensionality];
		int index = 0;
		for (int i = 0; i < size(); i++) {
			float weight = weights.get(i);
			for (int j = 0; j < ensembleBuffer[i].length; j++) {
				newVector[index] = weight * ensembleBuffer[i][j];
				index++;
			}
		}

		return new DenseVector(newVector);
	}

	/**
	 * @return The ranks of the spaces (a rank for each projection function)
	 *         representing the linearized examples
	 */
	public float[] getRanks() {
		float[] ranks = new float[size()];

		for (int i = 0; i < size(); i++) {
			ranks[i] = get(i).getRank();
		}
		return ranks;
	}

	@Override
	public Vector getLinearRepresentation(Example example) {
		try {
			return getDenseVectorByEnsembleAndJuxtaposition(example);
		} catch (InstantiationException e) {
			e.printStackTrace();
			return null;
		}
	}

	@Override
	public Example getLinearizedExample(Example example, String representationName) {

		Vector denseVector = getLinearRepresentation(example);

		HashMap<String, Representation> representations = new HashMap<String, Representation>();
		representations.put(representationName, denseVector);

		return new SimpleExample(example.getLabels(), representations);
	}

	@Override
	public SimpleDataset getLinearizedDataset(Dataset dataset, String representationName) {
		SimpleDataset resDataset = new SimpleDataset();

		int count = 1;
		for (Example e : dataset.getExamples()) {
			resDataset.addExample(getLinearizedExample(e, representationName));
			if (count % 100 == 0) {
				logger.info("Projected " + count + " examples.");
			}
			count++;
		}

		return resDataset;
	}

	@Override
	public int getEmbeddingSize() {
		int res = 0;
		for (NystromMethod nystromMethod : this) {
			res += nystromMethod.getEmbeddingSize();
		}
		return res;
	}

}
