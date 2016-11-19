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
            weight[i] = Math.random();
        }
    }

    private double activationF(double x) {
        return 1 / (1 + Math.pow(Math.E, -x)); //sigmoid function
    }

    public void countSign(double[] input) {
        double sigma = 0; //hitung sigma
        for (int i = 0; i < nInput; i++) {
            sigma += input[i] * weight[i];
        }
        sign = activationF(sigma); //masukkan kedalam AF
    }

    public void countErrSingle(int target) {
        error = target - sign;
    }

    public void countErrOut(double target) {
        error = sign * (1 - sign) * (target - sign); //ikutin rumus
    }

    public void countErrHid(double sum) {
        error = sign * (1 - sign) * sum; //ikutin rumus
    }

    public void updateWeight(double learnRate, double[] input) {
        for (int i = 0; i < nInput; i++) { //semua input bobotnya diatur ulang
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