import java.io.Serializable;

class Neuron implements Serializable {
    private double sign;
    private double error;
    private final int nInput; //jumlah input yang masuk termasuk bias
    private final double[] weight;

    public Neuron(int nInput) {
        this.nInput = nInput;
        weight = new double[nInput];
        for (int i = 0; i < nInput; i++) {
            weight[i] = Math.random(); //random initial weight
        }
    }

    private double activationF(double x) {
        return 1 / (1 + Math.pow(Math.E, -x)); //sigmoid
    }

    public void countSign(double[] input) {
        double sigma = 0;
        for (int i = 0; i < nInput; i++) {
            sigma += input[i] * weight[i];
        }
        sign = activationF(sigma);
    }

    public void countErrSingle(int target) {
        error = target - sign;
    }

    public void countErrOut(int target) {
        error = sign * (1 - sign) * (target - sign);
    }

    public void countErrHid(double sum) {
        error = sign * (1 - sign) * sum;
    }

    public void updateWeight(double learnRate, double[] input) {
        for (int i = 0; i < nInput; i++) {
            weight[i] += (learnRate * error * input[i]);
        }
    }

    public double getSign() {
        return sign;
    }

    public double getError() {
        return error;
    }

    public double[] getWeight() {
        return weight;
    }
}