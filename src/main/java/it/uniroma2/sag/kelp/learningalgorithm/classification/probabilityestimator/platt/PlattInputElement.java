package it.uniroma2.sag.kelp.learningalgorithm.classification.probabilityestimator.platt;

public class PlattInputElement {

	private int label;
	private float value;

	public PlattInputElement(int label, float value) {
		super();
		this.label = label;
		this.value = value;
	}

	public int getLabel() {
		return label;
	}

	public float getValue() {
		return value;
	}
}
