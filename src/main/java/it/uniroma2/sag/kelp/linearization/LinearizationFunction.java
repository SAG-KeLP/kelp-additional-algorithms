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

package it.uniroma2.sag.kelp.linearization;

import it.uniroma2.sag.kelp.data.dataset.Dataset;
import it.uniroma2.sag.kelp.data.dataset.SimpleDataset;
import it.uniroma2.sag.kelp.data.example.Example;
import it.uniroma2.sag.kelp.data.representation.Vector;

/**
 * This interface allows implementing function to linearized examples through
 * linear representations, i.e. vectors
 * 
 * 
 * @author Danilo Croce
 * 
 */
public interface LinearizationFunction {

	/**
	 * Given an input <code>Example</code>, this method generates a linear
	 * <code>Representation></code>, i.e. a <code>Vector</code>.
	 * 
	 * @param example
	 *            The input example.
	 * @return The linearized representation of the input example.
	 */
	public Vector getLinearRepresentation(Example example);

	/**
	 * This method linearizes an input example, providing a new example
	 * containing only a representation with a specific name, provided as input.
	 * The produced example inherits the labels of the input example.
	 * 
	 * @param example
	 *            The input example.
	 * @param vectorName
	 *            The name of the linear representation inside the new example
	 * @return
	 */
	public Example getLinearizedExample(Example example, String representationName);

	/**
	 * This method linearizes all the examples in the input <code>dataset</code>
	 * , generating a corresponding linearized dataset. The produced examples
	 * inherit the labels of the corresponding input examples.
	 * 
	 * @param dataset
	 *            The input dataset
	 * @param representationName
	 *            The name of the linear representation inside the new examples
	 * @return
	 */
	public SimpleDataset getLinearizedDataset(Dataset dataset, String representationName);

	/**
	 * @return the size of the resulting embedding, i.e. the number of resulting
	 *         vector dimensions
	 */
	public int getEmbeddingSize();

}
