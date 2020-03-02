using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Security.Cryptography.X509Certificates;
using NeuralNetwork.Scripts.Data;
using NeuralNetwork.Scripts.NetToolbox;
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
        [SerializeField] private double[] sequenceIndexor;
        
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
        public ENetworkImplementation NetworkFunction;
        public enum ENetworkImplementation
        {
            DataBasePrediction,
            GameEntityControl,
        }
        public NeuralNetController Controller;
        public bool inputStreamOn;
        public double[] ExternalInputs;
        public double[] OutputToExternal;
        
        #endregion
        
        #region Initialisation
        public void InitializeTraining(NeuralNetworkManager.ERunningMode eRunningMode, NeuralNetworkManager neuralNetworkManager,  int epochs = 0, int instanceID = 0, NetData netData = null)
        {
            Debug.Log("Starting Initialisation");
            _NetData = netData;
            NeuralNetworkManager = neuralNetworkManager;
            InstanceID = instanceID;
            
            // Setting Up InputsList and OutputsList for Controllers
            if (ExternalInputs.Length == 0)
            {
                ExternalInputs = new double[inputLayerConstructor.Neurons];
            }
            if (OutputToExternal.Length == 0)
            {
                OutputToExternal = new double[outputLayerConstructor.Neurons];
               
            }
            // Setting Up Network For Training
            if (NeuralNetworkManager.runningMode == NeuralNetwork.NeuralNetworkManager.ERunningMode.Train)
            {
               if(NeuralNetworkManager.NewTraining) InitializeNetwork(NeuralNetworkManager.InitialWeightsDelta);
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

        private void InitializeNetwork(double initWeightsDelta)
        {
            Debug.Log("Initializing Network");
            // Retrieving Network Construction Values And Adding 'X' Hidden Layers =====================================
            numInput = inputLayerConstructor.Neurons;
            for (int i = 0; i < hiddenLayersConstructor.Count; i++)
            {
                numHidden.Add(hiddenLayersConstructor[i].Neurons);
            }

            numOutput = outputLayerConstructor.Neurons;
            
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
                            h_Outputs.Add(new double[hiddenLayersConstructor[i].Neurons]);
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
                            h_Outputs.Add(new double[hiddenLayersConstructor[i].Neurons]);
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
                            h_Outputs.Add(new double[hiddenLayersConstructor[i].Neurons]);  // <= ADDED ON DEBUG 
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
                
                InitializeWeights(WeightsCount(), initWeightsDelta);
                
                
                
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

        private void InitializeWeights(int nbr, double initWeightsDelta)
        {
            random = new Random(0);
            double[] initialWeights = new  double[nbr];
            double lo = -initWeightsDelta; // -0.01
            double hi = initWeightsDelta;
            for (int i = 0; i < initialWeights.Length; ++i)
                initialWeights[i] = (hi - lo) * random.NextDouble() + lo;
            _NetData.InstanceWeights = initialWeights;
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
            int weightcount = WeightsCount();
            sequenceIndexor = CreateIndexor(weightcount);
            //**********************************************
            //double[] inputTest = new double[inputs.Length];
            //inputTest[0] = 0;
            //inputTest[1] = 1;
            //inputTest[2] = 2;
            //outputs = ComputeOutputs(inputTest, numHidden.Count);
            //DisplayOutputStrings(outputs);
            //**********************************************
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
                            if (j == L_h_hWeights[i].Length - 1)
                            {
                                for (int l = 0; l < h_Biases[i+1].Length; l++)
                                {
                                    result[k++] = h_Biases[i+1][l];
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
                            if (j == L_h_hWeights[i].Length - 1)
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
        private double[] ComputeOutputs(double[] xValues)
        {
            if (xValues.Length != numInput)
                throw new Exception("Bad xValues array length");

            int hiddenLayersCount = numHidden.Count;
            double[] outputSums = new double[outputLayerConstructor.Neurons];
            
            for (int i = 0; i < inputs.Length; i++)
            {
                inputs[i] = xValues[i];
            }
            
            // ONE HIDDEN LAYER NETWORK ================================================================================
            if (hiddenLayersCount==1)
            {
                // Input To First Hidden
                int hid = numHidden[0];
                double[] hiddenSums = new double[hid];
                for (int i = 0; i < hid; i++)
                {
                    for (int j = 0; j < inputLayerConstructor.Neurons; j++)
                    {
                        hiddenSums[i] += inputs[j] * i_hWeights[j][i]; // Calculate Input*Weights on Input to Hidden layer
                    }
                }
                for (int i = 0; i < hid; i++)
                {
                    hiddenSums[i] += h_Biases[0][i];
                }

                for (int i = 0; i < hid; i++)
                {
                    h_Outputs[0][i] =
                        ActivatorFunctionDouble(hiddenLayersConstructor[0].ActivatorFunction, hiddenSums[i], false, numInput);
                }

                for (int i = 0; i < numOutput; i++)
                {
                    for (int j = 0; j < numHidden[0]; j++)
                    {
                        outputSums[i] += h_Outputs[0][j] * L_h_hWeights[0][j][i];
                    }
                }

                for (int i = 0; i < numOutput; i++)
                {
                    outputSums[i] += o_Biases[i];
                }
                // Case SoftMax => We want all the outputs to be computed as an unique sample
                if (outputLayerConstructor.ActivatorFunction == ActivatorType.Softmax)
                {
                    double[] softMaxOutput = SoftmaxActivator(outputSums, false);
                    Array.Copy(softMaxOutput, outputs, softMaxOutput.Length);
                    Debug.Log("Output Softmax Activator");
                }
                else
                {
                    for (int i = 0; i < numOutput; i++)
                    {
                        outputSums[i] = ActivatorFunctionDouble(outputLayerConstructor.ActivatorFunction, outputSums[i], false, numHidden[0]);
                    }
                    outputs = outputSums;
                    Debug.Log("Output Generic Analog Activator");

                }

                return outputs;
            }
            // MULTIPLE HIDDEN LAYERS NETWORK ================================================================================
            if (hiddenLayersCount > 1)
            {
                for (int H = 0; H < hiddenLayersCount; H++)
                {
                    double[] hiddensSums = new double[numHidden[H]];
                    // Input To Hidden 0
                    if (H == 0)
                    {
                        for (int i = 0; i < numHidden[H]; i++)
                        {
                            for (int j = 0; j < inputLayerConstructor.Neurons; j++)
                            {
                                hiddensSums[i] +=
                                    inputs[j] * i_hWeights[j][i]; // Calculate Input*Weights on Input to Hidden layer
                            }
                        }

                        for (int i = 0; i < numHidden[H]; i++)
                        {
                            hiddensSums[i] += h_Biases[H][i];
                        }

                        for (int i = 0; i < numHidden[H]; i++)
                        {
                            h_Outputs[H][i] = ActivatorFunctionDouble(hiddenLayersConstructor[H].ActivatorFunction,
                                hiddensSums[i], false, numInput);
                        }
                    }

                    // Hidden To Hidden
                    if (H < hiddenLayersCount - 1)
                    {
                        hiddensSums = new double[numHidden[H+1]];
                        for (int i = 0; i < numHidden[H + 1]; i++)
                        {
                            for (int j = 0; j < numHidden[H]; j++)
                            {
                                hiddensSums[i] += h_Outputs[H][j] * L_h_hWeights[H][j][i];
                            }
                        }

                        for (int i = 0; i < numHidden[H + 1]; i++)
                        {
                            hiddensSums[i] += h_Biases[H + 1][i];
                        }

                        for (int i = 0; i < numHidden[H + 1]; i++)
                        {
                            h_Outputs[H + 1][i] = ActivatorFunctionDouble(hiddenLayersConstructor[H+1].ActivatorFunction,
                                hiddensSums[i], false, numHidden[H]);
                        }
                    }

                    // Hidden To Output
                    if (H == hiddenLayersCount - 1)
                    {
                        for (int i = 0; i < numOutput; i++)
                        {
                            for (int j = 0; j < numHidden[H]; j++)
                            {
                                outputSums[i] += h_Outputs[H][j] * L_h_hWeights[H][j][i];
                            }
                        }

                        for (int i = 0; i < numOutput; i++)
                        {
                            outputSums[i] += o_Biases[i];
                        }
                    }
                }

                // Case SoftMax => We want all the outputs to be computed as an unique sample
                if (outputLayerConstructor.ActivatorFunction == ActivatorType.Softmax)
                {
                    double[] softMaxOutput = SoftmaxActivator(outputSums, false);
                    //outputs = new double[softMaxOutput.Length];
                    Array.Copy(softMaxOutput, outputs, softMaxOutput.Length);
                    Debug.Log("Output Softmax Activator");

                }
                else
                {
                    for (int i = 0; i < numOutput; i++)
                    {
                        outputSums[i] = ActivatorFunctionDouble(outputLayerConstructor.ActivatorFunction, outputSums[i], false, numHidden[numHidden.Count-1]);
                    }

                    outputs = outputSums;
                    Debug.Log("Output Generic Analog Activator");

                }

                return outputs;
            }

            Debug.Log("Sample Loop Executed"); 
            return outputs;
        }

        private void BackPropagateGradient(double[] tValues, double trainingRate, double momentum, double weightDecay = 1)
        {
            // Les tValues sont les valeurs voulues. La base de donnée de learning doit être labelisée => Label = tValue pur chaque sample de données
            // le trainingRate est le coefficient 
            
            
        }
        
        
        #endregion
        
        #region Utils
        private static double[] CreateIndexor(int lenght)
        {
            double[] indexor = new double[lenght];
            for (int i = 0; i < lenght; i++)
            {
                indexor[i] = i;
            }
            return indexor;
        }
        private static void DisplayOutputStrings(double[] outputs)
        {
            string outputValues = "";
            for (int i = 0; i < outputs.Length; i++)
            {
                outputValues += outputs[i].ToString();
            }
        }
        private static double ActivatorFunctionDouble(ActivatorType activatorType, double entry, bool isDerivative, int prevNeurons = 1)
        {
            if (!isDerivative)
            {
                switch (activatorType)
                {
                    case ActivatorType.Identity :
                        entry = IdentityActivator.CalculateValue(entry);
                        break;
                    case ActivatorType.Sigmoid :
                        entry = SigmoidActivator.CalculateValue(entry);
                        break;
                    case ActivatorType.Relu :
                        entry = ReluActivator.CalculateValue(entry);
                        break;
                    case ActivatorType.Tanh :
                        entry = TanhActivator.CalculateValue(entry);
                        break;
                    case ActivatorType.Average:
                        entry = AverageActivator.CalculateValue(entry, prevNeurons);
                        break; 
                }
            }

            if (isDerivative)
            {
                switch (activatorType)
                {
                    case ActivatorType.Identity :
                        entry = IdentityActivator.CalculateDerivative(entry);
                        break;
                    case ActivatorType.Sigmoid :
                        entry = SigmoidActivator.CalculateDerivative(entry);
                        break;
                    case ActivatorType.Relu :
                        entry = ReluActivator.CalculateDerivative(entry);
                        break;
                    case ActivatorType.Tanh :
                        entry = TanhActivator.CalculateDerivative(entry);
                        break;
                    case ActivatorType.Average:
                        entry = AverageActivator.CalculateValue(entry, prevNeurons);
                        break; 
                }
            }
            

            return entry;
        }
        private static double[] SoftmaxActivator(double[] entry, bool isDerivative, double[]error = null)
        {
            double[] result = new double[entry.Length];
            if (isDerivative)
            {
                result = Scripts.NetToolbox.SoftmaxActivator.CalculateDerivative(entry, error);
            }

            if (!isDerivative)
            {
                result = Scripts.NetToolbox.SoftmaxActivator.CalculateValue(entry);

            }

            return result;
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
        
        private double MeanCrossEntropyError(double[][] trainData)
        {
            double sumError = 0.0;
            double[] xValues = new double[numInput];
            double[] tValues = new double[numOutput];

            for (int i = 0; i < trainData.Length; ++i)
            {
                Array.Copy(trainData[i], xValues, numInput); // get inputs 
                Array.Copy(trainData[i], numInput, tValues, 0, numOutput); // get targets
                double[] yValues = this.ComputeOutputs(xValues); // compute outputs
                for (int j = 0; j < numOutput; ++j)
                {
                    sumError += Math.Log(yValues[j]) * tValues[j]; // CE error
                }
            }
            return -1.0 * sumError / trainData.Length;
        }
        private double MeanSquaredError(double[][] trainData) // used as a training stopping condition
        {
            // average squared error per training tuple
            double sumSquaredError = 0.0;
            double[] xValues = new double[numInput]; // first numInput values in trainData
            double[] tValues = new double[numOutput]; // last numOutput values

            for (int i = 0; i < trainData.Length; ++i) 
            {
                // walk thru each training case. looks like (6.9 3.2 5.7 2.3) (0 0 1)
                //  where the parens are not really there
                Array.Copy(trainData[i], xValues, numInput); // get xValues.
                Array.Copy(trainData[i], numInput, tValues, 0, numOutput); // get target values
                double[] yValues = this.ComputeOutputs(xValues); // compute output using current weights
                for (int j = 0; j < numOutput; ++j)
                {
                    double err = tValues[j] - yValues[j];
                    sumSquaredError += err * err;
                }
            }

            return sumSquaredError / trainData.Length;
        }
        #endregion
        
        #region Genetic
        public void Genetic_OnInstanceEnd(List<NetLossParameter> paramatersForEvaluation) // Triggers Only With Genetic Learning
        {
            Debug.Log("InstanceEnd");
            if (NeuralNetworkManager.runningMode == NeuralNetworkManager.ERunningMode.Train)
            {
                Genetic_ComputeLossFunction(paramatersForEvaluation);
            }
            if (NeuralNetworkManager.runningMode == NeuralNetworkManager.ERunningMode.Execute)
            {
                NeuralNetworkManager.Genetic_ForceStartNextEpoch();
            }
            
        }
        public void Genetic_ComputeLossFunction(
            List<NetLossParameter> errorParameters)
        {
            // Compare Values to Actual DNA evaluation parameters values
            double performanceIndex = 0; // so we need to set-up a performance value wich will compare values from ActualBestDna parameters to this instance parameters
            List<NetLossParameter> actualDnaSolvers = new List<NetLossParameter>();
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
            if (eRunningMode == NeuralNetworkManager.ERunningMode.Train)
            {
                if (DNAHasUpgrade || forceInstanceDNAReset) // Getting NetData Best DNA, Randomizing by trainingRate and Reinject in Instance
                {
                    SetWeightsAndBiases(Genetic_RandomizeWeightsAndBiasesFromData(netData.InstanceWeights, NeuralNetworkManager.TrainingRate));
                    Debug.Log("Set DNA from Data");
                }
                else // Getting DNA, Randomizing and Reinject
                {
                    SetWeightsAndBiases(Genetic_RandomizeWeightsAndBiasesFromData(_NetData.InstanceWeights, NeuralNetworkManager.TrainingRate));
                }
            }
            if(!gameObject.activeSelf) gameObject.SetActive(true);
                // if no DNA upgrade
            Controller.InstanceReset();
            
        }
        private double[] Genetic_RandomizeWeightsAndBiasesFromData(double[] dataWeights, double TrainingRate)
        {
            random = new Random(0);
            double[] randomizedWeights = new  double[dataWeights.Length];
            randomizedWeights = dataWeights.ToArray();
            double lo = -TrainingRate;
            double hi = TrainingRate;
            for (int i = 0; i < randomizedWeights.Length; ++i)
                randomizedWeights[i] += (hi - lo) * random.NextDouble() + lo;
            _NetData.InstanceWeights = randomizedWeights;
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
                InitializeWeights(WeightsNumber, NeuralNetworkManager.InitialWeightsDelta);
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
                inputStreamOn = true;
               
            }
            if (eRunningMode == NeuralNetworkManager.ERunningMode.Execute)
            {
                IsTraining = false;
                IsExecuting = true;
                inputStreamOn = true;
                
            }
        }

        public void UseInstance(double[] entryValues)
        {
            ComputeOutputs(entryValues);
            OutputToExternal = outputs;
        }
        #endregion
      

    }
}