using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Security.Cryptography.X509Certificates;
using NeuralNetwork.Scripts.Data;
using NeuralNetwork.Utils;
using UnityEngine;
using Random = System.Random;


namespace NeuralNetwork
{
   
    public class NeuralNet : MonoBehaviour
    {
        [Header("Neural Network Architecture")]
        
        [HideInInspector] 
        public NeuralNetworkManager NeuralNetworkManager;
        
        public NetLayerConstructor inputLayerConstructor = new NetLayerConstructor();
        public List<NetLayerConstructor> hiddenLayersConstructor = new List<NetLayerConstructor>();
        public NetLayerConstructor outputLayerConstructor = new NetLayerConstructor();

        public int WeightsNumber;

        public static Random random;

        private int numInput = 0;
        private List<int> numHidden = new List<int>();
        private int numOutput = 0;
        
        private double[] inputs;
        
        private double[][] i_hWeights; // First Hidden Layer : input to Hidden1
        private List<double[][]> h_hWeights = new List<double[][]>(); //Middle Hidden Layer    // Last Hidden Layer : hidden to output
        private List<double[]> h_Biases = new List<double[]>(); 
        private List<double[]> h_Outputs = new List<double[]>();
      
        private double[] o_Biases;

        private double[] outputs;

        // Gradients de Back-Propagation 
        private double[] oGrads; // output gradients for back-propagation
        private double[] hGrads; // hidden gradients for back-propagation

        // Momentums de Back-Propagation
        private double[][] ihPrevWeightsDelta;  
        private List<double[]> hPrevBiasesDelta = new List<double[]>();
        private double[][] hoPrevWeightsDelta;
        private double[] oPrevBiasesDelta;
        
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
        
        #region Initialisation
        public void InitializeNeuralNetwork(NeuralNetworkManager.ERunningMode eRunningMode, int epochs = 0, int instanceID = 0, NeuralNetworkManager neuralNetworkManager = null, NetData netData = null)
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
            if (NeuralNetworkManager.runningMode == NeuralNetwork.NeuralNetworkManager.ERunningMode.Train && NeuralNetworkManager.NewTraining)
            {
               if(NeuralNetworkManager.NewTraining) InitializeNetwork();
               if(!NeuralNetworkManager.NewTraining) SetWeightsAndBiasesFromData(_NetData, NeuralNetworkManager.LearningLogic, NeuralNetworkManager.TrainingRate);

            }
            if (NeuralNetworkManager.runningMode == NeuralNetwork.NeuralNetworkManager.ERunningMode.Execute)
            {
                SetWeightsAndBiasesFromData(_NetData, NeuralNetworkManager.LearningLogic, NeuralNetworkManager.TrainingRate);
            }
            SetInstanceRunningMode(eRunningMode);
           
        }
        #endregion
      
        #region Network_Management
        private void InitializeNetwork()
        {
            numInput = inputLayerConstructor.Neurons;
            for (int i = 0; i < hiddenLayersConstructor.Count; i++)
            {
                numHidden.Add(hiddenLayersConstructor[i].Neurons);
            }
            numOutput = outputLayerConstructor.Neurons;
            
            inputs = new double[inputLayerConstructor.Neurons];
            i_hWeights = MakeMatrix(inputLayerConstructor.Neurons, hiddenLayersConstructor[0].Neurons);
            for (int i = 0; i < hiddenLayersConstructor.Count; i++)
            {
                if (hiddenLayersConstructor.Count == 1 || hiddenLayersConstructor.Count > 1 && i == hiddenLayersConstructor.Count - 1)
                {
                    h_hWeights.Add(MakeMatrix(hiddenLayersConstructor[i].Neurons, outputLayerConstructor.Neurons));
                    o_Biases = new double[outputLayerConstructor.Neurons];
                    outputs = new double[outputLayerConstructor.Neurons];
                }
                if (hiddenLayersConstructor.Count > 1 && i == hiddenLayersConstructor.Count - 2)
                {
                    h_hWeights.Add(MakeMatrix(hiddenLayersConstructor[i].Neurons, hiddenLayersConstructor[i + 1].Neurons));
                    h_Biases.Add(new double[hiddenLayersConstructor[i+1].Neurons]);
                    h_Outputs.Add(new double[hiddenLayersConstructor[i+1].Neurons]);
                }
            }
            o_Biases = new double[outputLayerConstructor.Neurons];
            outputs = new double[outputLayerConstructor.Neurons];
            InitializeWeights(WeightsCount());
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
            if (hiddensCount == 1)
            {
                nbr += hiddenLayersConstructor[0].Neurons * hiddenLayersConstructor[1].Neurons;
                nbr += hiddenLayersConstructor[1].Neurons * outputLayerConstructor.Neurons;
            }
            if (hiddensCount > 1)
            {
                for (int i = 0; i < hiddensCount-2; i++)
                {
                    nbr += hiddenLayersConstructor[i].Neurons * hiddenLayersConstructor[i + 1].Neurons;
                }
                nbr += hiddenLayersConstructor[hiddensCount - 2].Neurons * outputLayerConstructor.Neurons;
            }
            nbr += hiddenLayersConstructor[hiddensCount - 1].Neurons + outputLayerConstructor.Neurons;
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
        private void SetWeightsAndBiases(double[] weights, double[] biases = null)
        {
            int k = 0;
            for (int i = 0; i < numInput; i++)
            {
                for (int j = 0; j < numHidden[0]; j++)
                {
                    i_hWeights[i][j] = weights[k++];
                }
            }
            for (int i = 0; i < numHidden[0]; i++)
            {
                h_Biases[0][i] = weights[k++];
            }
        }
        
        private double[] GetWeightsAndBiases(NeuralNet instanceNetwork, out List<double> _weights, out List<double> _biases)
        {
            double[] result = new double[numWeights];
            int k = 0;
            for (int i = 0; i < ihWeights.Length; ++i)
            for (int j = 0; j < ihWeights[0].Length; ++j)
                result[k++] = ihWeights[i][j];
            for (int i = 0; i < hBiases.Length; ++i)
                result[k++] = hBiases[i];
            for (int i = 0; i < hoWeights.Length; ++i)
            for (int j = 0; j < hoWeights[0].Length; ++j)
                result[k++] = hoWeights[i][j];
            for (int i = 0; i < oBiases.Length; ++i)
                result[k++] = oBiases[i];
            return result;
            return weights;
        }
        

        private double[] ExecuteSequence(double[] inputs)
        {
            double[] outputs = new double[OutputToExternal.Count];
            double computedOutput = 0;
            for (int i = 0; i < inputs.Length; i++)
            {
                
            }
                    
                    
                    
                    
                    
            return outputs;
        }

        
        
        
        
        
        
        
        
        
        #endregion
        #region Genetic
        
        public void SetWeightsAndBiasesFromData(NetData netData, NeuralNetworkManager.ELearningLogic learningLogic, double trainingRate)
        {
            if (netData.HasData)
            {
                SetWeightsAndBiases(netData.InstanceWeights, netData.InstanceBiases);
            }
            else
            {
                InitializeWeights();
            }

        }
        
        public void Genetic_GetInstanceWeightsAndBiases(NeuralNet instanceNetwork, out List<double> _weights, out List<double> _biases)
        {
            List<double> biases = new List<double>();
            List<double> weights = new List<double>();
            for (int i = 0; i < instanceNetwork.inputLayerConstructor.NeuronsInLayer.Count; i++)
            {
                weights.Add(instanceNetwork.inputLayerConstructor.NeuronsInLayer[i].Weights[0]);
            }
            biases.Add(instanceNetwork.inputLayerConstructor.LayerBias);
            for (int i = 0; i < instanceNetwork.hiddenLayersConstructor.Count; i++)
            {
                for (int j = 0; j < instanceNetwork.hiddenLayersConstructor[i].NeuronsInLayer.Count; j++)
                {
                    for (int k = 0; k < instanceNetwork.hiddenLayersConstructor[i].NeuronsInLayer[j].PreviousNeurons; k++)
                    {
                        weights.Add(instanceNetwork.hiddenLayersConstructor[i].NeuronsInLayer[j].Weights[k]);
                    }
                }
                biases.Add(instanceNetwork.hiddenLayersConstructor[i].LayerBias);
            }
            
            for (int j = 0; j < instanceNetwork.outputLayerConstructor.NeuronsInLayer.Count; j++)
            {
                for (int k = 0; k < instanceNetwork.outputLayerConstructor.NeuronsInLayer[j].PreviousNeurons; k++)
                {
                    weights.Add(instanceNetwork.outputLayerConstructor.NeuronsInLayer[j].Weights[k]);
                }
            }
            biases.Add(instanceNetwork.outputLayerConstructor.LayerBias);
            
            _weights = weights;
            _biases = biases;
        } 
        
       
       
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
        private List<double> RandomizeWeightsAndBiasesFromData(List<double> dataWeights, double TrainingRate)
        {
            List<double> randomized = new List<double>();
            for (int i = 0; i < dataWeights.Count; i++)
            {
                var result = Random.Range((float) dataWeights[i] - (float) TrainingRate, (float) dataWeights[i] + (float) TrainingRate);
                randomized.Add(result);
            }
            
            return randomized;
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

        #region Others
        
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