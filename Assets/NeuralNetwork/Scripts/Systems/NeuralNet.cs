using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Security.Cryptography.X509Certificates;
using NeuralNetwork.Scripts.Data;
using NeuralNetwork.Scripts.NeuronActivatorFunctions;
using NeuralNetwork.Utils;
using UnityEngine;
using Random = System.Random;


namespace NeuralNetwork
{
   
    public class NeuralNet : MonoBehaviour
    {
        #region Fields
        [Header("Neural Network Architecture")]
        
        [HideInInspector] 
        public NeuralNetworkManager NeuralNetworkManager;
        
        public NetLayerConstructor inputLayerConstructor = new NetLayerConstructor();
        public List<NetLayerConstructor> hiddenLayersConstructor = new List<NetLayerConstructor>();
        public NetLayerConstructor outputLayerConstructor = new NetLayerConstructor();

        /// <summary>
        /// Network ====================================================================================================
        /// </summary>
        public double[] ResultArrayTest;
        
        //==============================================================================================================
        public int WeightsNumber;

        private static Random random;

        private int numInput = 0;
        private List<int> numHidden = new List<int>();
        private int numOutput = 0;
        
        private double[] inputs;
        
        [SerializeField] private double[][] i_hWeights; // First Hidden Layer : input to Hidden1
        [SerializeField] private List<double[][]> L_h_hWeights = new List<double[][]>(); //Middle Hidden Layer    // Last Hidden Layer : hidden to output
        [SerializeField] private List<double[]> h_Biases = new List<double[]>(); 
        [SerializeField] private List<double[]> h_Outputs = new List<double[]>();
        [SerializeField] private double[] o_Biases;

        [SerializeField] private double[] outputs;

        // Gradients de Back-Propagation 
        [SerializeField] private double[] oGrads; // output gradients for back-propagation
        [SerializeField] private List<double[]> hGrads = new List<double[]>(); // hidden gradients for back-propagation

        // Weight Decay
        private double weightDecay = 0.0001;
        // Momentums de Back-Propagation
        [SerializeField] private double momentum = 0.01;
        [SerializeField] private List<double[][]> hPrevWeightsDelta = new List<double[][]>();  
        [SerializeField] private List<double[]> hPrevBiasesDelta = new List<double[]>();
        [SerializeField] private double[][] oPrevWeightsDelta;
        [SerializeField] private double[] oPrevBiasesDelta;

        
        //==============================================================================================================
        [HideInInspector] public NetData _NetData;
        
        [Header("Network Execution")]
        public int InstanceID;
        public bool IsExecuting;
        public bool IsTraining;
     
        [Header("Input From World and Output To World")]
        public NeuralNetController Controller;
        public bool inputStreamOn;
        public List<double> ExternalInputs = new List<double>();
        public List<double> OutputToExternal = new List<double>();
        
        #endregion
        
        #region Initialisation
        public void InitializeTraining(NeuralNetworkManager.ERunningMode eRunningMode, NeuralNetworkManager neuralNetworkManager,  int epochs = 0, int instanceID = 0, NetData netData = null)
        {
            Debug.Log("Starting Initialisation");
            _NetData = netData;
            NeuralNetworkManager = neuralNetworkManager;
            InstanceID = instanceID;
            // Setting Up InputsList and OutputsList for Controllers
            if (ExternalInputs.Count == 0)
            {
                for (int i = 0; i <  inputLayerConstructor.Neurons; i++)
                {
                    ExternalInputs.Add(0);
                }
            }
            if (OutputToExternal.Count == 0)
            {
                for (int i = 0; i <  outputLayerConstructor.Neurons; i++)
                {
                   OutputToExternal.Add(0);
                }
            }
            // Setting Up Network For Training
            if (NeuralNetworkManager.runningMode == NeuralNetwork.NeuralNetworkManager.ERunningMode.Train)
            {
               if(NeuralNetworkManager.NewTraining) InitializeNetwork();
               if(!NeuralNetworkManager.NewTraining) SetWeightsAndBiasesFromData(_NetData, NeuralNetworkManager.LearningLogic, NeuralNetworkManager.TrainingRate);

            }
            // Setting Up Network For Execution
            if (NeuralNetworkManager.runningMode == NeuralNetwork.NeuralNetworkManager.ERunningMode.Execute)
            {
                SetWeightsAndBiasesFromData(_NetData, NeuralNetworkManager.LearningLogic, NeuralNetworkManager.TrainingRate);
            }
            // Setting Up This Running Mode
            SetInstanceRunningMode(eRunningMode);
           
        }
        #endregion
      
        #region Network_Management

        private void InitializeNetwork()
        {
            // Retrieving Network Construction Values And Adding 'X' Hidden Layers =====================================
            numInput = inputLayerConstructor.Neurons;
            inputLayerConstructor.Activator = ActivatorFactory.Produce(inputLayerConstructor.ActivatorFunction);
            for (int i = 0; i < hiddenLayersConstructor.Count; i++)
            {
                numHidden.Add(hiddenLayersConstructor[i].Neurons);
                hiddenLayersConstructor[i].Activator =
                    ActivatorFactory.Produce(hiddenLayersConstructor[i].ActivatorFunction);
            }

            numOutput = outputLayerConstructor.Neurons;
            outputLayerConstructor.Activator = ActivatorFactory.Produce(outputLayerConstructor.ActivatorFunction);
            
            // Initialize Arrays And Matrix ============================================================================
            inputs = new double[inputLayerConstructor.Neurons];
            i_hWeights = MakeMatrix(inputLayerConstructor.Neurons, hiddenLayersConstructor[0].Neurons);
            // If Network contains only one layer ======================================================================
            if (hiddenLayersConstructor.Count == 1)
            {
                L_h_hWeights.Add(MakeMatrix(hiddenLayersConstructor[0].Neurons, outputLayerConstructor.Neurons));
                h_Biases.Add(new double[hiddenLayersConstructor[0].Neurons]);
                h_Outputs.Add(new double[hiddenLayersConstructor[0].Neurons]);
                o_Biases = new double[outputLayerConstructor.Neurons];
                outputs = new double[outputLayerConstructor.Neurons];
                // Gradients
                hGrads.Add(new double[hiddenLayersConstructor[0].Neurons]);
                oGrads = new double[outputLayerConstructor.Neurons];
                // Weights Delta
                hPrevWeightsDelta.Add(MakeMatrix(inputLayerConstructor.Neurons, hiddenLayersConstructor[0].Neurons));
                hPrevBiasesDelta.Add(new double[hiddenLayersConstructor[0].Neurons]);
                oPrevWeightsDelta = MakeMatrix(hiddenLayersConstructor[0].Neurons, outputLayerConstructor.Neurons);
                oPrevBiasesDelta = new double[outputLayerConstructor.Neurons];
            }
            // If Network Contains more than one layer =================================================================
            {
                if (hiddenLayersConstructor.Count > 1)
                {
                    for (int i = 0; i < hiddenLayersConstructor.Count; i++)
                    {
                        if (i == 0)
                        {
                            Debug.Log("i=0");
                            L_h_hWeights.Add(MakeMatrix(hiddenLayersConstructor[i].Neurons,
                                hiddenLayersConstructor[i + 1].Neurons));
                            h_Biases.Add(new double[hiddenLayersConstructor[i].Neurons]);
                            h_Outputs.Add(new double[hiddenLayersConstructor[i + 1].Neurons]);
                            // Gradients
                            hGrads.Add(new double[hiddenLayersConstructor[i].Neurons]);
                            // Weights Delta
                            hPrevWeightsDelta.Add(MakeMatrix(inputLayerConstructor.Neurons,
                                hiddenLayersConstructor[i].Neurons));
                            hPrevBiasesDelta.Add(new double[hiddenLayersConstructor[i].Neurons]);
                        }

                        // => Setting up hidden layer i neurons and hidden layer i+1 neurons and biases
                        if (i > 0 && i < hiddenLayersConstructor.Count - 1)
                        {
                            Debug.Log("i>=0 && i < hiddenLayersContructor.Count");

                            L_h_hWeights.Add(MakeMatrix(hiddenLayersConstructor[i].Neurons,
                                hiddenLayersConstructor[i + 1].Neurons));
                            h_Biases.Add(new double[hiddenLayersConstructor[i].Neurons]);
                            h_Outputs.Add(new double[hiddenLayersConstructor[i + 1].Neurons]);
                            // Gradients
                            hGrads.Add(new double[hiddenLayersConstructor[i].Neurons]);
                            // Weights Delta
                            hPrevWeightsDelta.Add(MakeMatrix(hiddenLayersConstructor[i].Neurons,
                                hiddenLayersConstructor[i + 1].Neurons));
                            hPrevBiasesDelta.Add(new double[hiddenLayersConstructor[i + 1].Neurons]);
                        }

                        // => Setting up last hidden layer neurons and output layer neurons and biases
                        if (i == hiddenLayersConstructor.Count - 1)
                        {
                            L_h_hWeights.Add(MakeMatrix(hiddenLayersConstructor[i].Neurons,  // h-o weights
                                outputLayerConstructor.Neurons));
                            h_Biases.Add(new double[hiddenLayersConstructor[i].Neurons]);
                            o_Biases = new double[outputLayerConstructor.Neurons];
                            //h_Outputs.Add(new double[outputLayerConstructor.Neurons]);
                            outputs = new double[outputLayerConstructor.Neurons];
                            // Gradients
                            oGrads = new double[outputLayerConstructor.Neurons];
                            // Weights Delta
                            oPrevWeightsDelta = MakeMatrix(hiddenLayersConstructor[i].Neurons,
                                outputLayerConstructor.Neurons);
                            oPrevBiasesDelta = new double[outputLayerConstructor.Neurons];
                        }
                    }
                }
                // Initialize Weights ======================================================================================
                
                //InitializeWeights(WeightsCount());
                int weightcount = WeightsCount();
                ResultArrayTest = new double[weightcount];
                for (int i = 0; i < WeightsCount(); i++)
                {
                    ResultArrayTest[i] = i;
                }
                SetWeightsAndBiases(ResultArrayTest);
            }
        }
        private static double[][] MakeMatrix(int rows, int cols) // helper for ctor
        {
            double[][] result = new double[rows][];
            for (int r = 0; r < result.Length; ++r)
                result[r] = new double[cols];
            return result;
        }
        private int WeightsCount()
        {
            int nbr = 0;
            int hiddensCount = hiddenLayersConstructor.Count;
            nbr = (inputLayerConstructor.Neurons * hiddenLayersConstructor[0].Neurons); //Input to first
            nbr += hiddenLayersConstructor[0].Neurons; //biases
            if (hiddensCount == 1)
            {
                nbr += hiddenLayersConstructor[0].Neurons * outputLayerConstructor.Neurons;
                nbr += outputLayerConstructor.Neurons; //biases
            }
            if (hiddensCount > 1)
            {
                for (int i = 0; i < hiddensCount; i++)
                {
                    if (i < hiddensCount - 1)
                    {
                        nbr += hiddenLayersConstructor[i].Neurons * hiddenLayersConstructor[i + 1].Neurons;
                        nbr += hiddenLayersConstructor[i+1].Neurons; 
                    }
                    if (i == hiddensCount - 1)
                    {
                        nbr += hiddenLayersConstructor[i].Neurons * outputLayerConstructor.Neurons;
                        nbr += outputLayerConstructor.Neurons;
                    }
                }
            }
            WeightsNumber = nbr;
            return nbr;
        }
        private void InitializeWeights(int nbr)
        {
            random = new Random(0);
            double[] initialWeights = new  double[nbr];
            double lo = -0.01;
            double hi = 0.01;
            for (int i = 0; i < initialWeights.Length; ++i)
                initialWeights[i] = (hi - lo) * random.NextDouble() + lo;
           SetWeightsAndBiases(initialWeights);
        }
        private void SetWeightsAndBiases(double[] weights)
        {
            int k = 0;
            int TestConnection = 0;
            for (int i = 0; i < numInput; i++)
            {
                for (int j = 0; j < numHidden[0]; j++) // layer I_h
                {
                    i_hWeights[i][j] = weights[k++];
                    TestConnection++;
                }
            }
            for (int i = 0; i < numHidden[0]; i++)
            {
                h_Biases[0][i] = weights[k++];
                TestConnection++;

            }
            // MULTIPLE HIDDENS ========================================================================================
            if (numHidden.Count > 1) 
            {
                for (int i = 0; i < numHidden.Count; i++) //Layers H_h
                {
                    if (i < numHidden.Count-1)
                    {
                        for (int j = 0; j < hiddenLayersConstructor[i].Neurons; j++) // ITERATE IN HIDDEN LAYERS LIST
                        {
                            for (int l = 0; l <hiddenLayersConstructor[i+1].Neurons; l++)
                            {
                                L_h_hWeights[i][j][l] = weights[k++];
                                TestConnection++;

                            }
                            if (j == hiddenLayersConstructor[i].Neurons-1)
                            {
                                for (int l = 0; l < hiddenLayersConstructor[i+1].Neurons; l++)
                                {
                                    h_Biases[i + 1][l] = weights[k++]; //h_biase[0] has been set
                                    TestConnection++;

                                }
                            }
                        }
                    }
                    if (i == numHidden.Count-1)
                    {
                        for (int j = 0; j < hiddenLayersConstructor[i].Neurons; j++) // ITERATE IN HIDDEN LAYERS LIST
                        {
                            for (int l = 0; l < numOutput; l++)
                            {
                                L_h_hWeights[i][j][l] = weights[k++];
                                TestConnection++;

                            }
                            if (j == hiddenLayersConstructor[i].Neurons-1)
                            {
                                for (int l = 0; l < outputLayerConstructor.Neurons; l++)
                                {
                                    o_Biases[l] = weights[k++];
                                    TestConnection++;

                                }
                            }
                        }
                    }
                }
            }
            // ONE HIDDEN ==============================================================================================
            if (numHidden.Count == 1)
            {
                for (int i = 0; i < numHidden[0]; i++)
                {
                    for (int j = 0; j < numOutput; j++)
                    {
                        L_h_hWeights[0][i][j] = weights[k++];
                    }
                }
                for (int i = 0; i < numOutput; i++)
                {
                    o_Biases[i] = weights[k++];
                }
            }
            // END =====================================================================================================
            Debug.Log(TestConnection);
            Debug.Log("Set" + ResultArrayTest.Length);
           ResultArrayTest = GetWeightsAndBiases();
        }
        public double[] GetWeightsAndBiases()
        {
            double[] result = new double[WeightsNumber];
            int k = 0;
            for (int i = 0; i < i_hWeights.Length; ++i)
            {
                for (int j = 0; j < i_hWeights[0].Length; ++j)
                    result[k++] = i_hWeights[i][j];
            }
            for (int i = 0; i < h_Biases[0].Length; ++i)
                result[k++] =  h_Biases[0][i];
            //--------------------------------------------
            if (numHidden.Count > 1)
            {
                for (int i = 0; i < L_h_hWeights.Count; ++i)
                {
                    if (i < L_h_hWeights.Count - 1)
                    {
                        for (int j = 0; j < L_h_hWeights[i].Length; j++)
                        {
                            for (int l = 0; l < L_h_hWeights[i][j].Length; l++)
                            {
                                result[k++] = L_h_hWeights[i][j][l];
                            }
                            if (j == L_h_hWeights[i][j].Length - 1)
                            {
                                for (int l = 0; l < h_Biases[i+1].Length; l++)
                                {
                                    result[k++] = h_Biases[i][l];
                                }
                            }
                        }
                    }

                    if (i == L_h_hWeights.Count - 1)
                    {
                        for (int j = 0; j < L_h_hWeights[i].Length; j++)
                        {
                            for (int l = 0; l < L_h_hWeights[i][j].Length; l++)
                            {
                                result[k++] = L_h_hWeights[i][j][l];
                            }
                            if (j == L_h_hWeights[i][j].Length - 1)
                            {
                                for (int l = 0; l < o_Biases.Length; l++)
                                {
                                    result[k++] = o_Biases[l];
                                }
                            }
                        }
                    }
                    
                }
            }

            if (numHidden.Count == 1)
            {
                for (int j = 0; j < L_h_hWeights[0].Length; j++)
                {
                    for (int l = 0; l < L_h_hWeights[0][j].Length; l++)
                    {
                        result[k++] = L_h_hWeights[0][j][l];
                    }

                }
                for (int l = 0; l < o_Biases.Length; l++)
                {
                    result[k++] = o_Biases[l];
                }
            }
            Debug.Log("Get => " + result);
            return result;
        }
        

        private double[] ExecuteSequence(double[] xValues)
        {
            if (xValues.Length != numInput)
                throw new Exception("Bad xValues array length");
            double[] outputs = new double[OutputToExternal.Count];
            double computedOutput = 0;
            for (int i = 0; i < inputs.Length; i++)
            {
                
            }
                    
                    
                    
                    
                    
            return outputs;
        }
        #endregion
        
        #region BackPropagation
        
        private void BackPropagation_OnSequenceEnd(double[] result)
        {
            for (int i = 0; i < result.Length; i++)
            {
                OutputToExternal[i] = result[i];
            }
            if (NeuralNetworkManager.runningMode == NeuralNetworkManager.ERunningMode.Train)
            {
                // BackPropagation Loop
            }
        }
        
        #endregion
        
        #region Genetic
        public void Genetic_OnInstanceEnd(List<NeuralNetworkPerformanceSolver> paramatersForEvaluation) // Triggers Only With Genetic Learning
        {
            if (NeuralNetworkManager.runningMode == NeuralNetworkManager.ERunningMode.Train)
            {
                Genetic_ComputePerformanceIndex(paramatersForEvaluation);
            }
            if (NeuralNetworkManager.runningMode == NeuralNetworkManager.ERunningMode.Execute)
            {
                NeuralNetworkManager.Genetic_ForceStartNextEpoch();
            }
            
        }
        public void Genetic_ComputePerformanceIndex(
            List<NeuralNetworkPerformanceSolver> errorParameters)
        {
            // Compare Values to Actual DNA evaluation parameters values
            double performanceIndex = 0; // so we need to set-up a performance value wich will compare values from ActualBestDna parameters to this instance parameters
            List<NeuralNetworkPerformanceSolver> actualDnaSolvers = new List<NeuralNetworkPerformanceSolver>();
            if (NeuralNetworkManager.ActualBestDNA.PerformanceSolvers.Count == errorParameters.Count)
            {
                actualDnaSolvers =
                    NeuralNetworkManager.ActualBestDNA.PerformanceSolvers; // getting the values from ActualBestDna
            }
            else
            {
                Debug.Log("NeuralNetworkManager Actual DNA has not been initialized. Error.");
            }
            for (int i = 0; i < errorParameters.Count; i++)
            {
                double perfIndOfi = 0;
                double actualDnaParam = actualDnaSolvers[i].EvaluationParameter;
                if (errorParameters[i].ParameterType == EParameterType.ShouldBeInferior
                ) // this instance value should be inferior as best dna value to consider it as better
                {
                        perfIndOfi = actualDnaParam -
                                     errorParameters[i].EvaluationParameter; // value will be positive
                        perfIndOfi *= errorParameters[i].EvaluationParameterWeight; // applying weight coeeficient
                        performanceIndex += perfIndOfi;
                    
                   
                }
                if (errorParameters[i].ParameterType == EParameterType.ShouldBeSuperior) // this instance value should be superior as best dna value to consider it as better
                {
                        perfIndOfi = errorParameters[i].EvaluationParameter -
                                     actualDnaParam;
                        perfIndOfi *= errorParameters[i].EvaluationParameterWeight;
                        performanceIndex += perfIndOfi;
                }
                if (errorParameters[i].ParameterType == EParameterType.ConvergeToExpectedValue)
                {
                    perfIndOfi = Mathf.Abs((float)errorParameters[i].EvaluationParameter - (float)errorParameters[i].ExpectedValue);
                    if (perfIndOfi < Mathf.Abs((float) actualDnaParam -
                                               (float) errorParameters[i].ExpectedValue))
                    {
                        performanceIndex += perfIndOfi;
                    }
                }
            }
            bool instanceHasBestDna = false;
            if (performanceIndex > 0)
            {
                instanceHasBestDna = true;
            }
            else
            {
                Debug.Log("Performance index is inferior than 0.");
            }
            Debug.Log("NeuralNet. Instance Ending");
            NeuralNetworkManager.Genetic_OnInstanceCompare(this, performanceIndex, errorParameters, instanceHasBestDna);

        }
        public void Genetic_NeuralNetRestart(NeuralNetworkManager.ERunningMode eRunningMode, NetData netData, bool DNAHasUpgrade, bool forceInstanceDNAReset)
        {
            if (NeuralNetworkManager.LearningLogic == NeuralNetworkManager.ELearningLogic.Genetic)
            {
                if (eRunningMode == NeuralNetworkManager.ERunningMode.Train && DNAHasUpgrade ||
                    eRunningMode == NeuralNetwork.NeuralNetworkManager.ERunningMode.Train && forceInstanceDNAReset)
                {

                    SetWeightsAndBiasesFromData(netData, NeuralNetworkManager.LearningLogic, NeuralNetworkManager.TrainingRate);
                    Debug.Log("Set DNA from Data");
                }
                if(!gameObject.activeSelf) gameObject.SetActive(true);
                // if no DNA upgrade
                Controller.InstanceReset();
            }
        }
        private double[] Genetic_RandomizeWeightsAndBiasesFromData(List<double> dataWeights, double TrainingRate)
        {
            random = new Random(0);
            double[] randomizedWeights = new  double[dataWeights.Count];
            double lo = TrainingRate;
            double hi = TrainingRate;
            for (int i = 0; i < randomizedWeights.Length; ++i)
                randomizedWeights[i] = (hi - lo) * random.NextDouble() + lo;
            return randomizedWeights;
        }
        
        #endregion
        
        #region Data
        public void SetWeightsAndBiasesFromData(NetData netData, NeuralNetworkManager.ELearningLogic learningLogic, double trainingRate)
        {
            if (netData.HasData)
            {
                SetWeightsAndBiases(netData.InstanceWeights.ToArray());
            }
            else
            {
                InitializeWeights(WeightsNumber);
            }

        }
        
        #endregion
        
        #region InstanceRunning
        
        private void SetInstanceRunningMode(NeuralNetworkManager.ERunningMode eRunningMode)
        {
            if (eRunningMode == NeuralNetworkManager.ERunningMode.Train)
            {
                IsTraining = true;
                IsExecuting = false;
            }
            if (eRunningMode == NeuralNetworkManager.ERunningMode.Execute)
            {
                IsTraining = false;
                IsExecuting = true;
            }
        }
        
        #endregion
      

    }
}