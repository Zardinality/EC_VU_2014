import java.io.* ;


public class Main {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		SphereEvaluation eva = new SphereEvaluation();
		player19 test = new player19();
		test.setSeed(0);
		test.setEvaluation(eva);
		test.run();
		Double result = test.getResult();
		System.out.println(Double.toString(result));
	}

}
