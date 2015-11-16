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

package it.uniroma2.sag.kelp.learningalgorithm.classification.pegasos;

import it.uniroma2.sag.kelp.data.dataset.Dataset;
import it.uniroma2.sag.kelp.data.example.Example;
import it.uniroma2.sag.kelp.data.label.Label;
import it.uniroma2.sag.kelp.data.representation.Vector;
import it.uniroma2.sag.kelp.learningalgorithm.BinaryLearningAlgorithm;
import it.uniroma2.sag.kelp.learningalgorithm.LinearMethod;
import it.uniroma2.sag.kelp.learningalgorithm.classification.ClassificationLearningAlgorithm;
import it.uniroma2.sag.kelp.predictionfunction.classifier.BinaryLinearClassifier;
import it.uniroma2.sag.kelp.predictionfunction.classifier.BinaryMarginClassifierOutput;
import it.uniroma2.sag.kelp.predictionfunction.model.BinaryLinearModel;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import com.fasterxml.jackson.annotation.JsonTypeName;

/**
 * It implements the Primal Estimated sub-GrAdient SOlver (PEGASOS) for SVM. It is a learning
 * algorithm for binary linear classification Support Vector Machines. It operates in an explicit
 * feature space (i.e. it does not relies on any kernel). Further details can be found in:<p>
 * 
 * [SingerICML2007] Y. Singer and N. Srebro. Pegasos: Primal estimated sub-gradient solver for SVM. 
 * In Proceeding of ICML 2007.
 * 
 * @author Simone Filice
 *
 */
@JsonTypeName("pegasos")
public class PegasosLearningAlgorithm implements LinearMethod, ClassificationLearningAlgorithm, BinaryLearningAlgorithm{

	private Label label;
	
	private BinaryLinearClassifier classifier;
	
	private int k = 1;
	private int iterations = 1000;
	private float lambda = 0.01f;
	
	private String representation;
	
	/**
	 * Returns the number of examples k that Pegasos exploits in its 
	 * mini-batch learning approach
	 * 
	 * @return k
	 */
	public int getK() {
		return k;
	}

	/**
	 * Sets the number of examples k that Pegasos exploits in its 
	 * mini-batch learning approach
	 * 
	 * @param k the k to set
	 */
	public void setK(int k) {
		this.k = k;
	}

	/**
	 * Returns the number of iterations
	 * 
	 * @return the number of iterations 
	 */
	public int getIterations() {
		return iterations;
	}

	/**
	 * Sets the number of iterations
	 * 
	 * @param T the number of iterations to set
	 */
	public void setIterations(int T) {
		this.iterations = T;
	}

	/**
	 * Returns the regularization coefficient
	 * 
	 * @return the lambda
	 */
	public float getLambda() {
		return lambda;
	}

	/**
	 * Sets the regularization coefficient
	 * 
	 * @param lambda the lambda to set
	 */
	public void setLambda(float lambda) {
		this.lambda = lambda;
	}
	
	public PegasosLearningAlgorithm(){
		this.classifier = new BinaryLinearClassifier();
		this.classifier.setModel(new BinaryLinearModel());
	}
	
	public PegasosLearningAlgorithm(int k, float lambda, int T, String Representation, Label label){
		this.classifier = new BinaryLinearClassifier();
		this.classifier.setModel(new BinaryLinearModel());
		this.setK(k);
		this.setLabel(label);
		this.setLambda(lambda);
		this.setRepresentation(Representation);
		this.setIterations(T);
	}
	
	@Override
	public String getRepresentation() {
		return representation;
	}

	@Override
	public void setRepresentation(String representation) {
		this.representation = representation;
		BinaryLinearModel model = this.classifier.getModel();
		model.setRepresentation(representation);
	}
	
	@Override
	public void learn(Dataset dataset) {		
		if(this.getPredictionFunction().getModel().getHyperplane()==null){			
			this.getPredictionFunction().getModel().setHyperplane(dataset.getZeroVector(representation));
		}

		for(int t=1;t<=iterations;t++){

			List<Example> A_t = dataset.getRandExamples(k);
			List<Example> A_tp = new ArrayList<Example>();
			List<Float> signA_tp = new ArrayList<Float>();
			float eta_t = ((float)1)/(lambda*t);
			Vector w_t = this.getPredictionFunction().getModel().getHyperplane();

			//creating A_tp
			for(Example example: A_t){
				BinaryMarginClassifierOutput prediction = this.classifier.predict(example);
				float y = -1;
				if(example.isExampleOf(label)){
					y=1;
				}

				if(prediction.getScore(label)*y<1){
					A_tp.add(example);
					signA_tp.add(y);
				}					
			}
			//creating w_(t+1/2)
			w_t.scale(1-eta_t*lambda);
			float miscassificationFactor = eta_t/k;
			for(int i=0; i<A_tp.size(); i++){
				Example example = A_tp.get(i);
				float y = signA_tp.get(i);
				this.getPredictionFunction().getModel().addExample(y*miscassificationFactor, example);
			}

			//creating w_(t+1)
			float factor = (float) (1.0/Math.sqrt(lambda)/Math.sqrt(w_t.getSquaredNorm()));
			if(factor < 1){
				w_t.scale(factor);
			}		
			
		}
				
	}

	@Override
	public PegasosLearningAlgorithm duplicate() {
		PegasosLearningAlgorithm copy = new PegasosLearningAlgorithm();
		copy.setK(k);
		copy.setLambda(lambda);
		copy.setIterations(iterations);
		copy.setRepresentation(representation);
		return copy;
	}

	@Override
	public void reset() {
		this.classifier.reset();		
	}

	@Override
	public BinaryLinearClassifier getPredictionFunction(){
		return this.classifier;
	}
	
	@Override
	public void setLabels(List<Label> labels){
		if(labels.size()!=1){
			throw new IllegalArgumentException("Pegasos algorithm is a binary method which can learn a single Label");
		}
		else{
			this.label=labels.get(0);
			this.classifier.setLabels(labels);
		}
	}


	@Override
	public List<Label> getLabels() {
		return Arrays.asList(label);
	}
	
	@Override
	public Label getLabel(){
		return this.label;
	}
	
	@Override
	public void setLabel(Label label){
		this.setLabels(Arrays.asList(label));
	}
}
