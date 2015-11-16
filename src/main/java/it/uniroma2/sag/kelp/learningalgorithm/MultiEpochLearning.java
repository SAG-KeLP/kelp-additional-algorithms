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

package it.uniroma2.sag.kelp.learningalgorithm;

import it.uniroma2.sag.kelp.data.dataset.Dataset;
import it.uniroma2.sag.kelp.data.label.Label;
import it.uniroma2.sag.kelp.predictionfunction.PredictionFunction;

import java.util.List;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonTypeName;

/**
 * It is a meta learning algorithms for online learning methods. It performs
 * multiple iterations on the training data
 * 
 * @author Simone Filice
 *
 */
@JsonTypeName("multiEpoch")
public class MultiEpochLearning implements MetaLearningAlgorithm{

	private LearningAlgorithm baseAlgorithm;
	private int epochs;

	public MultiEpochLearning(){
		
	}
	
	public MultiEpochLearning(int epochs, LearningAlgorithm baseAlgorithm, List<Label> labels){
		this.setEpochs(epochs);
		this.setBaseAlgorithm(baseAlgorithm);
		this.setLabels(labels);
	}
	
	@Override
	public void setBaseAlgorithm(LearningAlgorithm baseAlgorithm) {
		this.baseAlgorithm=baseAlgorithm;		
	}

	@Override
	public LearningAlgorithm getBaseAlgorithm() {
		return this.baseAlgorithm;
	}

	/**
	 * @return the number of epochs
	 */
	public int getEpochs() {
		return epochs;
	}

	/**
	 * @param epochs the number of epochs to set
	 */
	public void setEpochs(int epochs) {
		this.epochs = epochs;
	}

	@Override
	public void learn(Dataset dataset) {
		
		for(int i=0; i<epochs; i++){
			Dataset shuffledData = dataset.getShuffledDataset();
			this.baseAlgorithm.learn(shuffledData);
		}
		
	}

	@Override
	public void setLabels(List<Label> labels){		
		this.baseAlgorithm.setLabels(labels);
	}

	@Override
	@JsonIgnore
	public List<Label> getLabels() {
		return this.baseAlgorithm.getLabels();
	}

	@Override
	public MultiEpochLearning duplicate() {
		MultiEpochLearning copy = new MultiEpochLearning();
		copy.epochs=epochs;
		copy.setBaseAlgorithm(baseAlgorithm.duplicate());
		return copy;
	}

	@Override
	public void reset() {
		this.baseAlgorithm.reset();		
	}

	@Override
	public PredictionFunction getPredictionFunction() {
		return this.baseAlgorithm.getPredictionFunction();
	}



}
