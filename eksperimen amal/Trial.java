package tubes2ai;

import java.util.Random;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;

public class Trial {
    String clsName;
    Instances data;
    Classifier cls;
    Evaluation eval;
    double pctCorrect;
    
    public Trial(Instances data, Classifier cls, String clsName) {
        this.data = data;
        this.cls = cls;
        this.clsName = clsName;
    }
    
    public void evaluate(Instances data) throws Exception {
        this.eval = new Evaluation(data);
        cls.buildClassifier(data);
        eval.crossValidateModel(cls, data, 10, new Random(1));
    }
    
    public void printStatus() {
        System.out.print("Cls: " + clsName);
        pctCorrect = eval.pctCorrect();
        System.out.println("\tPctCorrect: " + pctCorrect +" %");
    }
}
