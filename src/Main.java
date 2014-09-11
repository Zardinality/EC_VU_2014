public class Main {

	public static void main(String[] args) {
		
		//SphereEvaluation eva = new SphereEvaluation();
		AckleyEvaluation eva = new AckleyEvaluation();
		player19 test = new player19();
		test.setSeed(2);
		test.setEvaluation(eva);
		test.run();
		Double result = test.getResult();
		System.out.println(Double.toString(result));
	}

}
