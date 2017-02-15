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

import it.uniroma2.sag.kelp.data.example.Example;
import it.uniroma2.sag.kelp.data.example.SequenceExample;
import it.uniroma2.sag.kelp.data.example.SequencePath;

import com.fasterxml.jackson.annotation.JsonTypeInfo;
import com.fasterxml.jackson.databind.annotation.JsonTypeIdResolver;

/**
 * A <code>SequenceExampleGenerator</code> generates a copy of an input
 * <code>Example</code> (reflecting an item in a <code>SequenceExample</code>)
 * enriched with information derived from the <Label>s assigned to the previous
 * <code>n</code> examples. <br>
 * This allows the <code>SequenceClassificationLearningAlgorithm</code> to learn
 * from the observations that are derived from a targeted example, but also from
 * its history, in terms of labels assigned to previous examples.
 * 
 * @author Danilo Croce
 *
 */
@JsonTypeInfo(use = JsonTypeInfo.Id.CUSTOM, include = JsonTypeInfo.As.PROPERTY, property = "sequenceExamplesGeneratorType")
@JsonTypeIdResolver(SequenceExampleGeneratorTypeResolver.class)
public interface SequenceExampleGenerator {

	/**
	 * At labeling time, this method allows to enrich a specific
	 * <code>Example</code> with the labels assigned by the classifier to the
	 * previous <code>Example</code>s
	 * 
	 * @param sequenceExample
	 *            The targeted sequence
	 * @param sequencePath
	 *            the sequence of <code>Label</code> assigned from a classifier
	 *            to the <code>SequenceExamlpe</code>
	 * @param offset
	 *            the offset of the targeted word in the sequence
	 * @return
	 */
	public Example generateExampleWithHistory(SequenceExample sequenceExample, SequencePath sequencePath, int offset);

	/**
	 * This method allows to enrich each <code>Example</code> from an input
	 * <code>SequenceExample</code> with the labels assigned by the classifier
	 * to the previous <code>Example</code>s
	 * 
	 * @param sequenceExample
	 *            The input sequence
	 * 
	 * @return
	 */
	public SequenceExample generateSequenceExampleEnrichedWithHistory(SequenceExample sequenceExample);

	/**
	 * @return the number <code>n</code> of elements (in the sequence) whose
	 *         labels are to be considered to enrich a targeted element
	 */
	public int getTransitionsOrder();
}
