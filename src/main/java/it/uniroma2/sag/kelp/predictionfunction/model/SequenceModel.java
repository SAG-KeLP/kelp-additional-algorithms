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

package it.uniroma2.sag.kelp.predictionfunction.model;

import it.uniroma2.sag.kelp.data.examplegenerator.SequenceExampleGenerator;
import it.uniroma2.sag.kelp.predictionfunction.PredictionFunction;

/**
 * This class implements a model produced by a
 * <code>SequenceClassificationLearningAlgorithm</code>
 * 
 * @author Danilo Croce
 *
 */
public class SequenceModel implements Model {

	/**
	 * The prediction function producing the emission scores to be considered in
	 * the Viterbi Decoding
	 */
	private PredictionFunction basePredictionFunction;

	private SequenceExampleGenerator sequenceExampleGenerator;

	public SequenceModel() {
		super();
	}

	public SequenceModel(PredictionFunction basePredictionFunction, SequenceExampleGenerator sequenceExampleGenerator) {
		super();
		this.basePredictionFunction = basePredictionFunction;
		this.sequenceExampleGenerator = sequenceExampleGenerator;
	}

	public PredictionFunction getBasePredictionFunction() {
		return basePredictionFunction;
	}

	public SequenceExampleGenerator getSequenceExampleGenerator() {
		return sequenceExampleGenerator;
	}

	@Override
	public void reset() {
	}

	public void setBasePredictionFunction(PredictionFunction basePredictionFunction) {
		this.basePredictionFunction = basePredictionFunction;
	}

	public void setSequenceExampleGenerator(SequenceExampleGenerator sequenceExampleGenerator) {
		this.sequenceExampleGenerator = sequenceExampleGenerator;
	}

}
