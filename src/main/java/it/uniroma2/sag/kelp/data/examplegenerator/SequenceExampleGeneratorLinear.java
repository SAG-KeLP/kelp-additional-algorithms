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

package it.uniroma2.sag.kelp.data.examplegenerator;

import com.fasterxml.jackson.annotation.JsonTypeName;

import it.uniroma2.sag.kelp.data.example.Example;
import it.uniroma2.sag.kelp.data.example.SequenceExample;
import it.uniroma2.sag.kelp.data.example.SequencePath;
import it.uniroma2.sag.kelp.data.example.SimpleExample;
import it.uniroma2.sag.kelp.data.representation.Representation;
import it.uniroma2.sag.kelp.data.representation.Vector;
import it.uniroma2.sag.kelp.data.representation.vector.SparseVector;

/**
 * A <code>SequenceExampleGeneratorLinearAlg</code> allows to <b>explicitly</b>
 * enrich a targeted <code>Example</code> (reflecting an item in a
 * <code>SequenceExample</code>) with information derived from the <Label>s
 * assigned to the previous <code>n</code> examples.
 * 
 * <br>
 * 
 * 
 * Given a representation used to represent an example, this class generates a
 * copy of an input <code>Example</code> enriched with additional features
 * reflecting the classes assigned to the previous examples in the sequence.
 * 
 * <br>
 * 
 * This class should be used when the learning algorithm used within the
 * <code>SequenceClassificationLearningAlgorithm</code> implements the
 * <code>LinearMethod</code> interface.
 * 
 * @author Danilo Croce
 *
 */
@JsonTypeName("se_gen_lin")
public class SequenceExampleGeneratorLinear implements SequenceExampleGenerator {

	/**
	 * The identifier of the representation used to represent an example in the
	 * sequence and which will be enriched
	 */
	private String originalRepresentationName;

	/**
	 * The number of examples preceding a target example to be considered during
	 * the manipulation process
	 */
	private int transitionsOrder;

	/**
	 * The weight to assign to each new feature added in the manipulation
	 * process
	 */
	private float transitionWeight;

	public SequenceExampleGeneratorLinear() {

	}

	/**
	 * @param transitionsOrder
	 *            The number of examples preceding a target example to be
	 *            considered during the manipulation process
	 * @param originalRepresentationName
	 *            The identifier of the representation used to represent an
	 *            example in the sequence and which will be enriched
	 * @param enrichedWithHistoryRepresentationName
	 *            The identifier of the new representation produced in the
	 *            manipulation process
	 * @param transitionWeight
	 *            The weight to assign to each new feature added in the
	 *            manipulation process
	 */
	public SequenceExampleGeneratorLinear(int transitionsOrder, String originalRepresentationName,
			float transitionWeight) {
		this.originalRepresentationName = originalRepresentationName;
		this.transitionsOrder = transitionsOrder;
		this.transitionWeight = transitionWeight;
	}

	public Example generateExampleWithHistory(SequenceExample sequenceExample, SequencePath p, int elementId) {

		Example innerExample = sequenceExample.getExample(elementId);
		Representation observationRepresentation = innerExample.getRepresentation(originalRepresentationName);

		Example enrichedObservedExample = new SimpleExample();

		if (transitionsOrder > 0) {
			String transitionString = p.getHistoryBefore(elementId, transitionsOrder);

			Representation enrichedObservationRepresentation = generateManipulatedRepresentation(
					observationRepresentation, transitionString);
			/*
			 * Enrich the observed representation with the previous transition
			 */
			enrichedObservedExample.addRepresentation(originalRepresentationName, enrichedObservationRepresentation);
		} else {
			enrichedObservedExample.addRepresentation(originalRepresentationName, observationRepresentation);
		}

		return enrichedObservedExample;
	}

	/**
	 * Given the representation of a targeted example and a string containing
	 * the sequence of labels assigned to the previous examples, this method
	 * produces a new representation with additional features reflecting the
	 * sequence of labels
	 * 
	 * @param representation
	 *            the name of the targeted representation
	 * @param transitionString
	 *            the string a string containing the sequence of labels assigned
	 *            to the previous examples
	 * @return the new enriched representation
	 */
	private Representation generateManipulatedRepresentation(Representation representation, String transitionString) {
		Representation newRepresentation = null;

		if (representation instanceof SparseVector) {
			try {
				newRepresentation = new SparseVector();
				newRepresentation.setDataFromText(
						representation.toString().trim() + " " + transitionString + ":" + transitionWeight);
				return newRepresentation;
			} catch (Exception e1) {
				e1.printStackTrace();
				return null;
			}
		} else {
			System.err.println("Warning: SequenceExampleGeneratorLinearAlg only work on SparseVector... now ");
			return null;
		}
	}

	public SequenceExample generateSequenceExampleEnrichedWithHistory(SequenceExample sequenceExample) {

		SequenceExample res = (SequenceExample) sequenceExample.duplicate();

		for (int elementId = 0; elementId < res.getLenght(); elementId++) {

			Example e = res.getExample(elementId);

			Representation newRepresentation = null;
			Vector vector = (Vector) e.getRepresentation(originalRepresentationName);
			if (transitionsOrder > 0) {
				String transitionString = new String();
				for (int j = elementId - transitionsOrder; j < elementId; j++) {
					if (j < 0) {
						transitionString += SequenceExample.SEQDELIM + j + "init";
					} else {
						Example ej = res.getExample(j);
						transitionString += SequenceExample.SEQDELIM + ej.getClassificationLabels().iterator().next();
					}
				}
				newRepresentation = generateManipulatedRepresentation(vector, transitionString);
			} else {
				newRepresentation = vector.copyVector();
			}

			e.addRepresentation(originalRepresentationName, newRepresentation);
		}

		return res;
	}

	/**
	 * @return The identifier of the representation used to represent an example
	 *         in the sequence and which will be enriched
	 */
	public String getRepresentationName() {
		return originalRepresentationName;
	}

	public int getTransitionsOrder() {
		return transitionsOrder;
	}

	/**
	 * @return The weight to assign to each new feature added in the
	 *         manipulation process
	 */
	public float getTransitionWeight() {
		return transitionWeight;
	}

	/**
	 * @param representationName
	 *            The identifier of the representation used to represent a
	 *            example in the sequence and which will be enriched
	 */
	public void setRepresentationName(String representationName) {
		this.originalRepresentationName = representationName;
	}

}
