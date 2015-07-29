/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package fastclassifier;


import fastclassifier.dataTypes.DataBoard;
import fastclassifier.dataTypes.FeatureDefinition;
import fastclassifier.dataTypes.SegmentedClassification;
import ace.datatypes.TrainedModel;
import fastclassifier.options.Parameters;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.core.Instances;

/**
 *
 * @author Bruno
 */
public class FastClassifier {

   
    
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        //example run.
        Controller control = Controller.getInstance();
        
        //double[] samples = control.readTxtFile("wolfcrab.txt");
        double[] samples = null;
        try {
            samples = control.wavToSamples("1.wav");
        } catch (Exception ex) {
            Logger.getLogger(FastClassifier.class.getName()).log(Level.SEVERE, null, ex);
        }
        System.out.println("samples read. "+ samples.length);
        
        //tests classification time
        final long startTime = System.currentTimeMillis();
        String clas = null;
        try {
            //classifies audio
            clas =classifyAudio(samples);
        } catch (Exception ex) {
            Logger.getLogger(FastClassifier.class.getName()).log(Level.SEVERE, null, ex);
        }
        
        
        final long endTime = System.currentTimeMillis();
        System.out.println(clas);
        System.out.println("Total execution time: " + (endTime - startTime) + "milliseconds" );
        
    }
    
    /**
     * categorizes audiofile according to choosen classifier
     * 
     * @param wavfile the wavfile to categorize
     * @return 
     */
    public static String classifyWav(String wavfile) throws Exception{
        Controller control = Controller.getInstance();
        
        double[] samples = control.wavToSamples(wavfile);
        double[][][] features = control.ExtractFeatures(samples, null);
        control.calcOverallFeatures(features);
        //the above method is required to run the below 2
        double[][] overallFeatures = control.getOverallFeatureValues();
        FeatureDefinition[] defs = control.getOverallFeatureDefinitions();
        //creates a databoard
        DataBoard db = control.createDataBoard(overallFeatures, defs);
        Instances inst = control.createInstances(db);
        TrainedModel model = null;
        try {
            model = control.createTrainedModel(Parameters.getClassifierPath());
        } catch (IOException ex) {
            Logger.getLogger(FastClassifier.class.getName()).log(Level.SEVERE, null, ex);
        } 
        
        SegmentedClassification[] classifications = control.classify(model,db,inst);
        
        //return only first classification
        return classifications[0].classifications[0];
        
    }
    
    /**
     * categorizes audiofile according to choosen classifier
     * 
     * @param wavfile the wavfile to categorize
     * @return 
     */
    public static String classifyAudio(double[]samples) throws Exception{
        Controller control = Controller.getInstance();
        
        double[][][] features = control.ExtractFeatures(samples, null);
        control.calcOverallFeatures(features);
        //the above method is required to run the below 2
        double[][] overallFeatures = control.getOverallFeatureValues();
        FeatureDefinition[] defs = control.getOverallFeatureDefinitions();
        //creates a databoard
        DataBoard db = control.createDataBoard(overallFeatures, defs);
        Instances inst = control.createInstances(db);
        TrainedModel model = null;
        try {
            model = control.createTrainedModel(Parameters.getClassifierPath());
        } catch (IOException ex) {
            Logger.getLogger(FastClassifier.class.getName()).log(Level.SEVERE, null, ex);
        } 
        
        SegmentedClassification[] classifications = control.classify(model,db,inst);
        
        //return only first classification
        return classifications[0].classifications[0];
        
    }
 
    
    //public static void storeInstances(DataBoard db,Instances)
    

     
}
