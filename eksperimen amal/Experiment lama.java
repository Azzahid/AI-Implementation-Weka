import java.io.FileOutputStream;
import java.io.ObjectOutputStream;
import java.util.Random;
import java.util.Scanner;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.bayes.NaiveBayes;

public class Experiment {
    
    private Instances inst;
    private Classifier cls;
    private Evaluation eval;
    private int best;            // masih pake n fold aja, ini nyimpan nilai n nya
    
    public Experiment(Instances inst, Classifier cls, Evaluation eval) {
        this.inst = inst;
        this.cls = cls;
        this.eval = eval;
    }
    
    public Experiment(Experiment exp) {
        this.inst = exp.inst;
        this.cls = exp.cls;
        this.eval = exp.eval;
    }
    
    public static void main(String[] args) throws Exception{
        String[] filenames;
        String path = "";
        filenames = new String[] {"Team.arff", "iris.arff", "mush.arff"};
        int nFile = filenames.length;
        Experiment[] exps = new Experiment[nFile];      // array of experiment. satu eksperimen = satu dataset
       
        //inisialisasi exps
        for (int i=0; i<nFile; i++){
            try {
                DataSource source = new DataSource(path + filenames[i]);
                Instances inst = source.getDataSet();
                Classifier cls = new NaiveBayes();
                inst.setClassIndex(inst.numAttributes()-1);
                cls.buildClassifier(inst);
                Evaluation eval = new Evaluation(inst);
                Experiment exp = new Experiment(inst, cls, eval);
                exps[i] = new Experiment(exp);
            } catch (Exception e) {
                System.out.println("errorrrr coy...");
            }
        }
        
        System.out.println("PRINT STATUS");
        System.out.println();
        
        int step = 10;           // lompatan percobaan fold nya
        
        for (int i=0; i<nFile; i++) {
            System.out.println(filenames[i] + "--------------------------------------------------");
            System.out.println("n fold and percentage of correct instance");
            int optimumFold=0;
            double maxPctCorrect = 0;
            int curFold = 5;
            
            // nyari optimum fold
            while (curFold<100) {
                double pctCorrect = exps[i].eval.pctCorrect();
                if (pctCorrect > maxPctCorrect) {
                    maxPctCorrect = pctCorrect;
                    optimumFold = curFold;
                }
                
                exps[i].eval.crossValidateModel(exps[i].cls, exps[i].inst, curFold, new Random(1));
                System.out.println(curFold + " CrossValidation: " + pctCorrect);
                curFold+=step;
            }
            exps[i].best = optimumFold;
            
            System.out.println("Max Percentage of Correct Instance: " + maxPctCorrect);
            System.out.println("Optimum Fold: " + optimumFold);
            System.out.print("Do you want to save optimum model (1/0): ");
            Scanner sc = new Scanner(System.in);
            String isSave = sc.next();
            
            try {
                if (Integer.parseInt(isSave)==1) {
                    exps[i].eval.crossValidateModel(exps[i].cls, exps[i].inst, exps[i].best, new Random(1));        // ini sebenarnya ngapain wkwk
                    System.out.print("input filename: ");
                    ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(sc.next()));               // save model
                    oos.writeObject(exps[i].cls);
                    oos.flush();
                }
            } catch (Exception e){
                System.err.println("error: " + e);
            }
           
            System.out.println();
        }
    }
}