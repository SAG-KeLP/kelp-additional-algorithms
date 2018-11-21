package it.uniroma2.sag.kelp.learningalgorithm.classification.probabilityestimator.platt;

import java.util.Vector;

public class PlattInputList {

	private Vector<PlattInputElement> list;
	private int positiveElement;
	private int negativeElement;

	public PlattInputList() {
		list = new Vector<PlattInputElement>();
	}

	public void add(PlattInputElement arg0) {
		if (arg0.getLabel() > 0)
			positiveElement++;
		else
			negativeElement++;

		list.add(arg0);
	}

	public PlattInputElement get(int index) {
		return list.get(index);
	}

	public int size() {
		return list.size();
	}

	public int getPositiveElement() {
		return positiveElement;
	}

	public int getNegativeElement() {
		return negativeElement;
	}

}
