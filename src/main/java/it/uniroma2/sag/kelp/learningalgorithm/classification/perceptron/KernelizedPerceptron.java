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

package it.uniroma2.sag.kelp.learningalgorithm.classification.perceptron;


import it.uniroma2.sag.kelp.data.label.Label;
import it.uniroma2.sag.kelp.kernel.Kernel;
import it.uniroma2.sag.kelp.learningalgorithm.KernelMethod;
import it.uniroma2.sag.kelp.predictionfunction.classifier.BinaryKernelMachineClassifier;
import it.uniroma2.sag.kelp.predictionfunction.model.BinaryKernelMachineModel;

import com.fasterxml.jackson.annotation.JsonTypeName;

/**
 * The perceptron learning algorithm algorithm for classification tasks (Kernel machine version). Reference:
 * <p> [Rosenblatt1957] F. Rosenblatt. The Perceptron – a perceiving and recognizing automaton. Report 85-460-1, Cornell Aeronautical Laboratory (1957)
 * 
 * @author Simone Filice
 *
 */
@JsonTypeName("kernelizedPerceptron")
public class KernelizedPerceptron extends Perceptron implements KernelMethod{


	private Kernel kernel;

	public KernelizedPerceptron(){
		this.classifier = new BinaryKernelMachineClassifier();
		this.classifier.setModel(new BinaryKernelMachineModel());
	}
	
	public KernelizedPerceptron(float alpha, float margin, boolean unbiased, Kernel kernel, Label label){
		this.classifier = new BinaryKernelMachineClassifier();
		this.classifier.setModel(new BinaryKernelMachineModel());
		this.setAlpha(alpha);
		this.setMargin(margin);
		this.setUnbiased(unbiased);
		this.setKernel(kernel);
		this.setLabel(label);
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
	public KernelizedPerceptron duplicate(){
		KernelizedPerceptron copy = new KernelizedPerceptron();		
		copy.setKernel(this.kernel);
		copy.setAlpha(this.alpha);
		copy.setMargin(this.margin);
		copy.setUnbiased(this.unbiased);		
		return copy;
	}
	
	@Override
	public BinaryKernelMachineClassifier getPredictionFunction(){
		return (BinaryKernelMachineClassifier) this.classifier;
	}


	
}
