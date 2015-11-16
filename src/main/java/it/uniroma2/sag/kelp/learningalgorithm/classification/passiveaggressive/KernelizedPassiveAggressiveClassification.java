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

package it.uniroma2.sag.kelp.learningalgorithm.classification.passiveaggressive;


import it.uniroma2.sag.kelp.data.label.Label;
import it.uniroma2.sag.kelp.kernel.Kernel;
import it.uniroma2.sag.kelp.learningalgorithm.KernelMethod;
import it.uniroma2.sag.kelp.predictionfunction.classifier.BinaryKernelMachineClassifier;
import it.uniroma2.sag.kelp.predictionfunction.model.BinaryKernelMachineModel;

import com.fasterxml.jackson.annotation.JsonTypeName;

/**
 * Online Passive-Aggressive Learning Algorithm for classification tasks (Kernel Machine version) .
 * Every time an example is misclassified it is added as support vector, with the weight that solves the 
 * passive aggressive minimization problem
 * 
 * reference: 
 * <p>
 * [CrammerJLMR2006] Koby Crammer, Ofer Dekel, Joseph Keshet, Shai Shalev-Shwartz and Yoram Singer
 * Online Passive-Aggressive Algorithms. Journal of Machine Learning Research (2006)
 * 
 * <p>The standard algorithm is modified, including the fairness extention from<p>
 * [FiliceECIR2014] S. Filice, G. Castellucci, D. Croce, and R. Basili. Effective Kernelized Online Learning 
 * in Language Processing Tasks. In collection of Advances in Information Retrieval, pp. 347–358, Springer International Publishing, 2014. 
 * 
 * 
 * @author      Simone Filice
 */

@JsonTypeName("kernelizedPA")
public class KernelizedPassiveAggressiveClassification extends PassiveAggressiveClassification implements KernelMethod{

	private Kernel kernel;

	public KernelizedPassiveAggressiveClassification(){
		this.classifier = new BinaryKernelMachineClassifier();
		this.classifier.setModel(new BinaryKernelMachineModel());
	}
	
	public KernelizedPassiveAggressiveClassification(float cp, float cn, Loss loss, Policy policy, Kernel kernel, Label label){
		this.classifier = new BinaryKernelMachineClassifier();
		this.classifier.setModel(new BinaryKernelMachineModel());
		this.setKernel(kernel);
		this.setLoss(loss);
		this.setCp(cp);
		this.setCn(cn);
		this.setLabel(label);	
		this.setPolicy(policy);
	}


	@Override
	public Kernel getKernel() {
		return kernel;
	}

	@Override
	public void setKernel(Kernel kernel) {
		this.kernel = kernel;
		this.getPredictionFunction().getModel().setKernel(kernel);
	}

		
	@Override
	public KernelizedPassiveAggressiveClassification duplicate(){
		KernelizedPassiveAggressiveClassification copy = new KernelizedPassiveAggressiveClassification();
		copy.setCp(this.cp);
		copy.setCn(c);
		copy.setFairness(this.fairness);
		copy.setKernel(this.kernel);
		copy.setLoss(this.loss);
		copy.setPolicy(this.policy);	
		//copy.setLabel(label);
		return copy;
	}

	@Override
	public BinaryKernelMachineClassifier getPredictionFunction(){
		return (BinaryKernelMachineClassifier) this.classifier;
	}
	
}
