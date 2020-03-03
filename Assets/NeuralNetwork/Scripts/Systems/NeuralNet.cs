using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Security.Cryptography.X509Certificates;
using NeuralNetwork.Scripts.Data;
using NeuralNetwork.Scripts.NetToolbox;
using NeuralNetwork.Utils;
using UnityEditor.VersionControl;
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
        [SerializeField] private int[] sequenceIndexor;
        
        //==============================================================================================================
        public int WeightsNumber;

        private static Random random;

        private int numInput = 0;
        private List<int> numHidden = new List<int>();
        private int numOutput = 0;
        
        private double[] inputs;
        
        private double[][] i_hWeights; // First Hidden Layer : input to Hidden1
        private List<double[][]> L_h_hWeights = new List<double[][]>(); //Middle Hidden Layer    // Last Hidden Layer : hidden to output
        private List<double[]> h_Biases = new List<double[]>(); 
        private List<double[]> h_Outputs = new List<double[]>();
        private double[] o_Biases;

        private double[] outputs;

        // Gradients de Back-Propagation 
        private double[] oGrads; // output gradients for back-propagation
        private List<double[]> hGrads = new List<double[]>(); // hidden gradients for back-propagation

        // Weight Decay
        private double weightDecay = 0.0001;
        // Momentums de Back-Propagation
        private double momentum = 0.01;
        private List<double[][]> hPrevWeightsDelta = new List<double[][]>();  
        private List<double[]> hPrevBiasesDelta = new List<double[]>();
        private double[][] oPrevWeightsDelta;
        private double[] oPrevBiasesDelta;

        
        //==============================================================================================================
        [HideInInspector] public NetData _NetData;
        
        [Header("Network Execution")]
        public int InstanceID;
        public bool IsExecuting;
        public bool IsTraining;

        public double LossFunctionResult;
            
        [Header("Input From World and Output To World")]
        public ENetworkImplementation NetworkFunction;
        public enum ENetworkImplementation
        {
            DataBasePrediction,
            GameEntityControl,
        }
        public NeuralNetController Controller;
        public bool inputStreamOn;// Allow the Controller to Use connected instance
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
                        ActivatorFunctions(hiddenLayersConstructor[0].ActivatorFunction, hiddenSums[i], false, numInput);
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
                        outputSums[i] = ActivatorFunctions(outputLayerConstructor.ActivatorFunction, outputSums[i], false, numHidden[0]);
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
                            h_Outputs[H][i] = ActivatorFunctions(hiddenLayersConstructor[H].ActivatorFunction,
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
                            h_Outputs[H + 1][i] = ActivatorFunctions(hiddenLayersConstructor[H+1].ActivatorFunction,
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
                        outputSums[i] = ActivatorFunctions(outputLayerConstructor.ActivatorFunction, outputSums[i], false, numHidden[numHidden.Count-1]);
                    }

                    outputs = outputSums;
                    Debug.Log("Output Generic Analog Activator");

                }

                return outputs;
            }

            Debug.Log("Sample Loop Executed"); 
            return outputs;
        }

        private void BackPropagateGradient(double[] tValues, double trainingRate, double momentum, double weightDecay = 0.001)
        {
            // Algorithme de Descente de gradient stochastique => voir Théorie Math pour comprendre.
            // Les tValues sont les valeurs voulues. La base de donnée de learning doit être labelisée => Label = tValue pur chaque sample de données
            // le trainingRate est le coefficient de modification des weights dans la fonction d'erreur
            // le weight decay (ou dégradation des pondérations) permet de limiter le sur-apprentissage en ajoutant une pénalié au résultat sur les weights de la fonction d'erreur
            if (tValues.Length != numOutput)
            {
                throw new Exception("Label target values doesn't fit output values length");
            }
            
            // Gradients de neurones de sortie ===========================================================================
            for (int i = 0; i < oGrads.Length; i++)
            {
                double derivative =
                    ActivatorFunctions(outputLayerConstructor.ActivatorFunction, outputs[i], true, numHidden[numHidden.Count-1]);
                oGrads[i] = derivative * (tValues[i] - outputs[i]);
            }
            
            // Gradients des neurones de couches hidden ==================================================================
            // Les gradients remontent dans le réseau, de la sortie vers l'entrée, donc on part de la dernière couche hidden
            int hiddenIndex = numHidden.Count-1;
            for (int i = hiddenIndex; i >= 0; i--)
            {
                if (i==hiddenIndex)
                {
                    for (int j = 0; j < hGrads[i].Length; j++)
                    {
                        double derivative = ActivatorFunctions(hiddenLayersConstructor[i].ActivatorFunction,
                            h_Outputs[i][j], true, numHidden[i]);
                        double sum = 0.0;
                        for (int k = 0; k < numOutput; k++)
                        {
                            double x = oGrads[k] * L_h_hWeights[i][j][k];
                            sum += x;
                        }
                        hGrads[i][j] = derivative * sum;
                    }
                }

                if (i < hiddenIndex && i >= 0)
                {
                    for (int j = 0; j < hGrads[i].Length; j++)
                    {
                        double derivative = ActivatorFunctions(hiddenLayersConstructor[i].ActivatorFunction,
                            h_Outputs[i][j], true, numHidden[i]);
                        double sum = 0.0;
                        for (int k = 0; k < numOutput; k++)
                        {
                            double x = hGrads[i+1][k] * L_h_hWeights[i][j][k];
                            sum += x;
                        }
                        hGrads[i][j] = derivative * sum;
                    }
                }
            }
            //============================================================================================================
            // Mise à jour des weights et de biais (pas d'ordre nécessaire pour la mise à jour du poids)
            for (int i = 0; i < i_hWeights.Length; i++)
            {
                for (int j = 0; j < i_hWeights[i].Length; j++)
                {
                    double delta = trainingRate * hGrads[0][j] * inputs[i]; 
                    i_hWeights[i][j] += delta;
                    i_hWeights[i][j] += momentum * hPrevWeightsDelta[0][i][j];
                    i_hWeights[i][j] -= (weightDecay * i_hWeights[i][j]);
                    hPrevWeightsDelta[0][i][j] = delta; 
                }
            }
            for (int i = 0; i < L_h_hWeights.Count; i++)
            {
                if (i < L_h_hWeights.Count - 1)
                {
                    for (int j = 0; j < L_h_hWeights[i].Length; j++)
                    {
                        for (int k = 0; k < L_h_hWeights[i][j].Length; k++)
                        {
                            double delta = trainingRate * hGrads[i + 1][k] * h_Outputs[j][k];
                            L_h_hWeights[i][j][k] += delta;
                            L_h_hWeights[i][j][k] += momentum * hPrevWeightsDelta[i + 1][j][k];
                            L_h_hWeights[i][j][k] -= (weightDecay * L_h_hWeights[i][j][k]);
                            hPrevWeightsDelta[i + 1][j][k] = delta;
                        }
                    }
                }

                if (i == L_h_hWeights.Count - 1)
                {
                    for (int j = 0; j < L_h_hWeights[i].Length; j++)
                    {
                        for (int k = 0; k < L_h_hWeights[i][j].Length; k++)
                        {
                            double delta = trainingRate * oGrads[k] * h_Outputs[j][k];
                            L_h_hWeights[i][j][k] += delta;
                            L_h_hWeights[i][j][k] += momentum * oPrevWeightsDelta[j][k];
                            L_h_hWeights[i][j][k] -= (weightDecay * L_h_hWeights[i][j][k]);
                            hPrevWeightsDelta[i + 1][j][k] = delta;
                        }
                    }
                }
                
            }

            for (int i = 0; i < h_Biases.Count; i++)
            {
                for (int j = 0; j < h_Biases[i].Length; j++)
                {
                    double delta = trainingRate * hGrads[i][j];
                    h_Biases[i][j] += delta;
                    h_Biases[i][j] += momentum * hPrevBiasesDelta[i][j];
                    h_Biases[i][j] -= weightDecay * h_Biases[i][j];
                    hPrevBiasesDelta[i][j] = delta;
                }
            }

            for (int i = 0; i < o_Biases.Length; i++)
            {
                double delta = trainingRate * oGrads[i];
                o_Biases[i] += delta;
                o_Biases[i] += momentum * oPrevBiasesDelta[i];
                o_Biases[i] -= weightDecay * o_Biases[i];
            }
           
            
        }
        
        #endregion
        
        #region Utils
        private static void Normalize(double[][] dataMatrix, int[] cols)
        {
            cols = CreateIndexor(cols.Length);
            // normalize specified cols by computing (x - mean) / sd for each value
            foreach (int col in cols)
            {
                double sum = 0.0;
                for (int i = 0; i < dataMatrix.Length; ++i)
                    sum += dataMatrix[i][col];
                double mean = sum / dataMatrix.Length;
                sum = 0.0;
                for (int i = 0; i < dataMatrix.Length; ++i)
                    sum += (dataMatrix[i][col] - mean) * (dataMatrix[i][col] - mean);
                // thanks to Dr. W. Winfrey, Concord Univ., for catching bug in original code
                double sd = Math.Sqrt(sum / (dataMatrix.Length - 1));
                for (int i = 0; i < dataMatrix.Length; ++i)
                    dataMatrix[i][col] = (dataMatrix[i][col] - mean) / sd;
            }
        }
        private static double[][] MakeMatrix(int rows, int cols) // helper for ctor
        {
            double[][] result = new double[rows][];
            for (int r = 0; r < result.Length; ++r)
                result[r] = new double[cols];
            return result;
        }
        private static int[] CreateIndexor(int lenght)
        {
            int[] indexor = new int[lenght];
            for (int i = 0; i < lenght; i++)
            {
                indexor[i] = i;
            }
            return indexor;
        }
        private static void ShuffleIndexor(int[] sequence)
        {
            for (int i = 0; i < sequence.Length; ++i)
            {
                int r = random.Next(i, sequence.Length);
                int tmp = sequence[r];
                sequence[r] = sequence[i];
                sequence[i] = tmp;
            }
        }
        private static int MaxIndex(double[] vector) // helper for Accuracy()
        {
            // index of largest value
            int bigIndex = 0;
            double biggestVal = vector[0];
            for (int i = 0; i < vector.Length; ++i)
            {
                if (vector[i] > biggestVal)
                {
                    biggestVal = vector[i]; bigIndex = i;
                }
            }
            return bigIndex;
        }

        private static void DisplayOutputStrings(double[] outputs)
        {
            string outputValues = "";
            for (int i = 0; i < outputs.Length; i++)
            {
                outputValues += outputs[i].ToString();
            }
        }
        private static double ActivatorFunctions(ActivatorType activatorType, double entry, bool isDerivative, int prevNeurons = 1)
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
                    case ActivatorType.Softmax : //Assuming Softmax derivative for Two Dimension events is the same as sigmoid.
                                                 //Softmax is a generalisation of Sigmoid on n Dimensions outputs
                                                 // See later for using the stronger Jacobian model  
                        entry = SigmoidActivator.CalculateDerivative(entry); 
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

        public void Train(double[][] allData, int maxEpochs, double learnRate, double momentum, double weightDecay, NeuralNetworkManager.ELossFunction lossFunction, int splitPurcentage)
        {
            int epoch = 0;
            
            Normalize(allData, new int[numInput]);
            
            double[] xValues = new double[numInput];
            double[] yValues = new double[numOutput];
            double[] tValues = new double[numOutput];
            // Splitting The AllData between trainData and testData
            Random rnd = new Random(0);
            int totRows = allData.Length;
            int numCols = allData[0].Length;

            int trainRows = (int)(totRows * (splitPurcentage/100)); // hard-coded 80-20 split
            int testRows = totRows - trainRows;

            double[][] trainData = new double[trainRows][];
            double[][]  testData = new double[testRows][];
            int[] sequence = new int[trainData.Length];
            sequence = CreateIndexor(sequence.Length);
            
            while (epoch < maxEpochs)
            {
                double mse_mcee = 0;
                if (lossFunction == NeuralNetworkManager.ELossFunction.MeanCrossEntropy)
                {
                    mse_mcee = MeanSquaredError(trainData);
                }
                if (lossFunction == NeuralNetworkManager.ELossFunction.MeanSquarredError)
                {
                    mse_mcee = MeanCrossEntropyError(trainData);
                }
                
                LossFunctionResult = mse_mcee;
                ShuffleIndexor(sequence);
                for (int i = 0; i < trainData.Length; i++)
                {
                    int index = sequence[i];
                    Array.Copy(trainData[index], xValues, numInput); // dans le tableau de Data : les valeurs de 0 à numInput => valeurs de test, celles de numInput à numOutput => label / tValues
                    Array.Copy(trainData[index], numInput, tValues, 0, numOutput); // On copie les valeurs du label à part.
                    ComputeOutputs(xValues);
                    BackPropagateGradient(tValues, learnRate, momentum, weightDecay);
                }
                epoch++;
            }

            if (epoch == maxEpochs)
            {
                Debug.Log("Training Sequence is done. Now Computing Accuracy Test");
                Array.Copy(allData, testData, allData[0].Length);
                double accuracy = Accuracy(testData);
                _NetData.PerformanceCoefficient = accuracy;
                BackPropagation_OnAccuracyComputed(accuracy);
            }
            
        }
       
        
        private void BackPropagation_OnAccuracyComputed(double result)
        {
           NeuralNetworkManager.BackPropagation_OnAccuracyResult(this, result);
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
        
        public double Accuracy(double[][] testData) 
        {
            // Pourcentage en utilisant la méthode "Plus haute Valeur Gagnante" dans les outputs.
            // Ici on ignora la valeur réelle des outputs, pour ne garder que son poids statistique. 
            int numCorrect = 0;
            int numWrong = 0;
            double[] xValues = new double[numInput]; // inputs
            double[] tValues = new double[numOutput]; // targets
            double[] yValues; // computed Y

            for (int i = 0; i < testData.Length; ++i)
            {
                Array.Copy(testData[i], xValues, numInput); // parse test data into x-values and t-values
                Array.Copy(testData[i], numInput, tValues, 0, numOutput);
                yValues = this.ComputeOutputs(xValues);
                int maxIndex = MaxIndex(yValues); // which cell in yValues has largest value?

                if (tValues[maxIndex] == 1.0) // ugly. consider AreEqual(double x, double y)
                    ++numCorrect;
                else
                    ++numWrong;
            }
            return (numCorrect * 1.0) / (numCorrect + numWrong); // ugly 2 - check for divide by zero
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
        public void UseInstance(double[] xValuesSimpleData = null, double[][] xValuesTrainData = null)
        {
            if(NetworkFunction == ENetworkImplementation.GameEntityControl) TrainGenetic(xValuesSimpleData);
            if(NetworkFunction == ENetworkImplementation.DataBasePrediction) TrainBackpropagate(xValuesTrainData);
        }
        private void TrainGenetic(double[] entryValues)
        {
            ComputeOutputs(entryValues);
            OutputToExternal = outputs;
        }

        private void TrainBackpropagate(double[][] entryValues)
        {
            
        }
        
        #endregion
      

    }
}