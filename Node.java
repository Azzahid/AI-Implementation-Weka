import java.io.Serializable;
import java.util.Arrays;

class Node implements Serializable {
    private double output; //nilai setelah keluar dari AF
    private double error;
    private final int nInput; //jumlah input yang masuk
    private final double[] weight;

    public Node(int nInput) {
        //tiap neuron punya bobot sebanyak inputnya (sudah termasuk bias)
        this.nInput = nInput;
        weight = new double[nInput];
        //untuk awal, semua bobot ditentukan bernilai 0
        Arrays.fill(weight, 0);
    }

    public Node(double[] predWeight) {
        //WARNING: constructor ini hanya digunakan saat load model
        //semua bobot sudah ditentukan nilainya
        weight = predWeight;
        nInput = weight.length;
    }

    private double activationF(double x) {
        //return x >= 0 ? 1 : 0; //step function dengan threshold = 0
        //return x >= 0 ? 1 : -1; //sign function
        return 1 / (1 + Math.pow(Math.E, -x)); //sigmoid function
    }

    public void countOutput(double[] input) {
        //hitung sigma x dikali w
        double sigma = 0;
        for (int i = 0; i < nInput; i++) {
            sigma += input[i] * weight[i];
        }
        //masukkan sigma kedalam AF
        output = activationF(sigma);
    }

    public void countErrSingle(int target, int sign) {
        //ATTENTION: perihal nilai sign (0 atau 1)
        //misal <o1, o2, o3> = <0.6, 0.2, 0.8>
        //jika di-"step" disini maka <s1,s2,s3> = <1,0,1>
        //    (bisa melanggar aturan, seharusnya hanya boleh ada satu angka 1)
        //jika di-"step" diluar maka <s1,s2,s3> = <0,0,1>
        //    (pasti hanya ada satu angka 1 yaitu yang nilai o terbesar)
        //sign = output > 0.5 ? 1 : 0; //hapus baris ini kalo mau "step"-an dari luar
        error = target - sign;
    }

    public void countErrOut(double target) {
        error = output * (1 - output) * (target - output); //ikutin rumus
    }

    public void countErrHid(double sum) {
        error = output * (1 - output) * sum; //ikutin rumus
    }

    public void updateWeight(double learnRate, double[] input) {
        //semua input bobotnya diatur ulang
        for (int i = 0; i < nInput; i++) {
            weight[i] += (learnRate * error * input[i]); //ikutin rumus
        }
    }

    public double getOutput() {
        return output;
    }

    public double getError() {
        return error;
    }

    public double[] getWeight() {
        return weight;
    }
}