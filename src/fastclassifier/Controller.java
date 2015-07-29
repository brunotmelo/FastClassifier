/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package fastclassifier;

import fastclassifier.Classification.InstanceClassifier;
import fastclassifier.GeneralTools.DSPMethods;
import fastclassifier.dataTypes.SegmentedClassification;
import ace.datatypes.TrainedModel;
import fastclassifier.dataTypes.DataBoard;
import fastclassifier.dataTypes.DataSet;
import fastclassifier.dataTypes.FeatureDefinition;
import fastclassifier.dataTypes.Taxonomy;
import fastclassifier.featureExtraction.Aggregators.Aggregator;
import fastclassifier.featureExtraction.FeatureProcessor;
import fastclassifier.featureExtraction.audioFeatures.*;
import fastclassifier.options.Parameters;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.Scanner;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import mckay.utilities.sound.sampled.AudioSamples;
import weka.core.Instances;

/**
 *
 * @author Bruno
 */
public class Controller {
    
    private FeatureExtractor[] features;
    private boolean[] definitions;
    private FeatureProcessor processor;
    private static Controller control;
    
    //return must be an array
    //make a documentation explaining which features represent each indice
    
    
    private Controller(){
        //TODO: user sets parameters
        //TODO: make processor settings be set separately. Example: set at launch to gain speed.
        //you will need to check if the parameters have been set before
        
        
        populateFeatures();
        int windowSize = Parameters.windowSize;
        double windowOverlap = Parameters.windowOverlap;
	double samplingRate = Parameters.samplingRate;
        //normalise as different function?
        boolean normalise = Parameters.normalise;
        boolean perWindowStats = Parameters.perWindowStats;
	boolean overallStats = Parameters.overallStats;
        
        try {
            //TODO: change definition input by the function one. To control which features to extract from the outside
            processor = new FeatureProcessor(windowSize,
                    windowOverlap, samplingRate, normalise, this.features,
                    this.definitions, perWindowStats,
                    overallStats);
        } catch (Exception ex) {
            Logger.getLogger(Controller.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
      
    public static Controller getInstance(){
        if(control == null){
            control = new Controller();
            return control;
        }else{
            return control;
        }
    }   
    
        /**
	 * Extract Features from audio File
	 *
	 * @param audioSamples
	 *            The samples of the audio(in mono) to be extracted features from.
	 * @param featuresToExtract
	 *            Array of booleans indicating which features to extract.
	 * @throws Exception
	 *             Throws an informative exception if the input parameters are
	 *             invalid.
	 */
    public double[][][] ExtractFeatures(double[] audioSamples, boolean[] featuresToExtract){
        
        
        
        //returns features for each window.
        
        double[][][] feat = null;
        try {
            feat = processor.extractFeatures(audioSamples, null);
        } catch (Exception ex) {
            Logger.getLogger(Controller.class.getName()).log(Level.SEVERE, null, ex);
        }
        
        return feat;
        
    }
    
    //exceptiontion is trown from getoverallrecordingfeatures
    public void calcOverallFeatures(double[][][] windowFeatureValues) throws Exception{
            //gets the aggregatorList
        Aggregator[] aggList = Parameters.getAggList();
//            Aggregator[] aggList = new Aggregator[2];
//            aggList[0] = new StandardDeviation();
//            aggList[1] = new Mean();
//			aggList[2] = new AreaMoments();
//			aggList[2].setParameters(new String[]{"MFCC"},new String[]{});
//			aggList[3] = new AreaMoments();
//			aggList[3].setParameters(new String[]{"LPC"},new String[]{});
//			aggList[4] = new AreaMoments();
//			aggList[4].setParameters(new String[]{"Derivative of MFCC"},new String[]{});
//			aggList[5] = new AreaMoments();
//			aggList[5].setParameters(new String[]{"Derivative of LPC"},new String[]{});
//			aggList[6] = new AreaMoments();
//			aggList[6].setParameters(new String[]{"Derivative of Method of Moments"},new String[]{});
//			aggList[7] = new AreaMoments();
//			aggList[7].setParameters(new String[]{"Method of Moments"},new String[]{});
//			aggList[8] = new AreaMoments();
//			aggList[8].setParameters(new String[]{"Area Method of Moments"},new String[]{});
//			aggList[9] = new AreaMoments();
//			aggList[9].setParameters(new String[]{"Derivative of Area Method of Moments"},new String[]{});
//			aggList[2] = new MFCC();
//			aggList[2] = new MultipleFeatureHistogram(new FeatureExtractor[]{new RMS(),new ZeroCrossings()},8);
//			aggList[3] = new MultipleFeatureHistogram(new FeatureExtractor[]{new MFCC()},4);
            processor.calcOverallRecordingFeatures(windowFeatureValues, aggList);
    }    
    
    public double[][] getOverallFeatureValues(){
        return processor.getOverallRecordingFeatureValues();
    }
    
    public FeatureDefinition[] getOverallFeatureDefinitions(){
        return processor.getOverallRecordingFeatureDefinitions();
    }
    
    
    
    //the first indice will be the feature. If you choose to get mean and stdeviation, even indecis will hold means and odd deviations.
    public double [][] getOverallFeaturesBackup(double[][][] windowFeatureValues){
        return processor.getOverallRecordingFeatures(
		windowFeatureValues, true,false);  
        
    } 
    
    public DataBoard createDataBoard(double[][] overallFeatures, FeatureDefinition[] defs){
       // defs[0] = new FeatureDefinition("MFCC Overall Average","MFCC calculations based upon Orange Cow code\n" +
//"This is the overall average over all windows.",true,13);
        
        DataSet[] values = new DataSet[1];
        
        values[0] = new DataSet();
        values[0].identifier = "FastClassifier";
        values[0].feature_values = overallFeatures;
        
        //String[] names = {"MFCC Overall Average"};
        String[] names = new String[defs.length];
        for(int i=0; i<defs.length; i++){
            names[i] = defs[i].name;
            //print names. Debug purposes
            //System.out.println(names[i]);
        }
        values[0].feature_names = names;
        
        Taxonomy tax = Parameters.getTaxonomy();
        DataBoard db = null;
        try {
            //create databoard
             db = new DataBoard(tax,defs,values,null);
            
            
        } catch (Exception ex) {
            Logger.getLogger(Controller.class.getName()).log(Level.SEVERE, null, ex);
        }
        
        return db;
 
    }
    
    /**
     * creates instances file. Needed to classify
     * @param db
     * @return 
     */
    public Instances createInstances(DataBoard db){
        Instances inst = null;
        try {
            //must be run after createdataboard
            inst = db.getInstanceAttributes("Testing Relation", 100);
            boolean use_top_level_features = true;
            //******this may lead to bugs (sub-sections)
            boolean use_sub_section_features = false;
            db.storeInstances(inst, use_top_level_features, use_sub_section_features);
            
            
        } catch (Exception ex) {
            Logger.getLogger(FastClassifier.class.getName()).log(Level.SEVERE, null, ex);
        }
        
        return inst;
    }
    
    /**
     * classifies the audio
     * @param model the classifier to be used
     * @param db the databoard to be used
     * @param inst the set of instances
     * @return the classifications
     */
    public SegmentedClassification[] classify(TrainedModel model, DataBoard db, Instances inst){
        try {
            return InstanceClassifier.classify(model, db, inst, null, false);
        } catch (Exception ex) {
            Logger.getLogger(FastClassifier.class.getName()).log(Level.SEVERE, null, ex);
        }
        return null;
    }    
    
    /**
     * loads a trained model from the path provided.
     * @param classifiers_file
     * @return
     * @throws IOException 
     */
    public TrainedModel createTrainedModel(String classifiers_file) throws IOException{
        TrainedModel model = null;
        
        try
                {
                    FileInputStream load_stream = new FileInputStream(new File(classifiers_file));
                    ObjectInputStream object_stream = new ObjectInputStream(load_stream);
                    model = (TrainedModel) object_stream.readObject();
                    load_stream.close();
                }
                catch (IOException e)
                {
                    throw new IOException("Invalid classifier file: " + classifiers_file);
                }catch (ClassNotFoundException e){
                    e.printStackTrace();
                }
        
        return model;
    }
    
    private void populateFeatures(){
        features = Parameters.getFeatures();
        definitions = Parameters.getDefinitions();
        
        }
    
    
    //get audio samples and return the normalised ones
    public double[] normaliseSamples(double[] audioSamples){
        return DSPMethods.normalizeSamples(audioSamples);
    }
    
    //method to be implemented
    private void splitWindows(){
        //split windows according to size
    }
    
    /**
     * reads the txt file indicated in path. I used it for testing 
     * with a double[] array set of samples saved as txt file.
     * @param path
     * @return double array contained inside the txt.
     */
    public static double[] readTxtFile(String path){
        try {
            Scanner in = new Scanner(new FileReader(path));
            ArrayList<Double> samples = new ArrayList<Double>();
            String next;
            while(in.hasNext()){
                next = in.next();
                samples.add(Double.parseDouble(next));
            }
            
            double[] primSamples = new double[samples.size()];
            
            for(int i = 0;i<samples.size(); i++){
                primSamples[i] = samples.get(i);
            }
            
            return primSamples;
            
        } catch (FileNotFoundException ex) {
            Logger.getLogger(Controller.class.getName()).log(Level.SEVERE, null, ex);
        }
        
        return null;
    }
    
    /**
	 * Returns the samples stored in the given audio file.
	 * <p>
	 * The samples are re-encoded using the sampling rate in the sampling_rate
	 * field. All channels are projected into one channel. Samples are
	 * normalised if the normalise field is true.
	 *
	 * @param path
	 *            The audio path to extract samples from.
         * @param 
	 * @return The processed audio samples. Values will fall between a minimum
	 *         of -1 and +1. The indice identifies the sample number.
	 * @throws Exception
	 *             An exception is thrown if a problem occurs during file
	 *             reading or pre- processing.
	 */
	public static double[] wavToSamples(String path) throws Exception {
		
                File recording_file = new File(path);
            
                boolean normalise = Parameters.normalise;
                double sampling_rate = Parameters.samplingRate;
                // Get the original audio and its format
		AudioInputStream original_stream = AudioSystem
				.getAudioInputStream(recording_file);
		AudioFormat original_format = original_stream.getFormat();

		// Set the bit depth
		int bit_depth = original_format.getSampleSizeInBits();
		if (bit_depth != 8 && bit_depth != 16)
			bit_depth = 16;

		// If the audio is not PCM signed big endian, then convert it to PCM
		// signed
		// This is particularly necessary when dealing with MP3s
		AudioInputStream second_stream = original_stream;
		if (original_format.getEncoding() != AudioFormat.Encoding.PCM_SIGNED
				|| original_format.isBigEndian() == false) {
			AudioFormat new_format = new AudioFormat(
					AudioFormat.Encoding.PCM_SIGNED, original_format
							.getSampleRate(), bit_depth, original_format
							.getChannels(), original_format.getChannels()
							* (bit_depth / 8), original_format.getSampleRate(),
					true);
			second_stream = AudioSystem.getAudioInputStream(new_format,
					original_stream);
		}

		// Convert to the set sampling rate, if it is not already at this
		// sampling rate.
		// Also, convert to an appropriate bit depth if necessary.
		AudioInputStream new_stream = second_stream;
		if (original_format.getSampleRate() != (float) sampling_rate
				|| bit_depth != original_format.getSampleSizeInBits()) {
			AudioFormat new_format = new AudioFormat(
					AudioFormat.Encoding.PCM_SIGNED, (float) sampling_rate,
					bit_depth, original_format.getChannels(), original_format
							.getChannels()
							* (bit_depth / 8), original_format.getSampleRate(),
					true);
			new_stream = AudioSystem.getAudioInputStream(new_format,
					second_stream);
		}

		// Extract data from the AudioInputStream
		AudioSamples audio_data = new AudioSamples(new_stream, recording_file
				.getPath(), false);

		// Normalise samples if this option has been requested
		if (normalise)
			audio_data.normalizeMixedDownSamples();

		// Return all channels compressed into one
		return audio_data.getSamplesMixedDown();
	}
}
