using System.Collections.Generic;
using System.IO;
using System.Linq;
using NeuralNetwork.Scripts.Data;
using NeuralNetwork.Utils;
using UnityEngine;

namespace NeuralNetwork
{
    public class NeuralNetworkManager : MonoBehaviour
    {
        #region References
        [Header("Instanciation")]
        public GameObject NetWorkPrefab;
        public string IANetworkName = "Neural Network Exemple";
       
        [Header("Instances")]
        [SerializeField] private NetData NetData;
        public string SaveNetDataFileName;
        
        [HideInInspector] public List<NeuralNet> NeuralNetworkInstances = new List<NeuralNet>();
        public NeuralNetworkEvaluate ActualBestDNA;
        [HideInInspector] public List<NeuralNetworkEvaluate> EvaluateDNAForInstancesIteration = new List<NeuralNetworkEvaluate>();
        public int InstancesEndedCount;
        
        [Header("Networks Management")] 
        public ENetworkFunction NetworkFunction;

        
        public ENetworkMode NetworkMode;
        public enum ENetworkMode
        {
            Train,
            Execute,
        }

        public EForceRandomization ForceGeneticsRandomization;
        public enum EForceRandomization
        {
            Yes,
            No,
        }

        public EAbsoluteValues AbsoluteValues;
        public enum EAbsoluteValues
        {
            Yes,
            No,
        }
        public enum ENetworkFunction
        {
            ComputeData,
            ControlEntity,
        }

       
        [SerializeField] public bool isNeuralNetTraining;
        [SerializeField] public bool isNeuralNetExecuting;
        
        [HideInInspector] public bool LoadFromBlueprint;
       

        [Header("Training Setup")] 
        public bool NewTraining = true;//check if instances is new or has already iterated
        public bool LoadFromJson;//will be removed after saving architecture refactoring
        public ETrainMode TrainMode;
        public enum ETrainMode
        {
            NewTraining,
            LoadFromJson,
        }

        public ESaveMode SaveMode;
        public enum ESaveMode
        {
            Override,
            New,
        }
         
        [Header("Training Options")]
        
        public double TrainingRate;//gradient d'update des weights
        public double MaxTrainingRate;
        public int Epochs;//Iterations of training
        public int TrainingBatchSize;//number of instances trained at the same time

        public bool AdjustTrainingRateAutomatically;
        [Range(0.01f, 50f)] public float TrainingRateChangePurcentage; //if NeuralNetwork can't upgrade for n = "TrainingRateEvolution", decrease Training Rate by this value
        public int TrainingRateEvolution;
        private bool DNAHasUpgrade;
        private bool ForceInstanceDNAReset; //if iteration don't get any upgrade avec n = TrainingRateEvolution epochs, reset on DNA data
        private int epochsWithoutDNAEvolutionCount;
        private List<double> previousIterationsCoefficients = new List<double>();
        [SerializeField] private double previousIterationsCoefficientAverage;
        
        public int EpochsCount;
        public EEvaluateIterationValueToActualDNA evaluateIterationValueToActualDna;
        public enum EEvaluateIterationValueToActualDNA
        {
            IterationResultsInferior,
            IterationResultsSuperior,
        }
        
        [Header("Neural Network Evaluation")] 
        public int DNAVersion;
        public double[] TrainingBestResults;
        public double[] InternalParameters;
        
        #endregion
        
        private void ResetNetData(NetData netData)
        {
            netData.HasData = false;
            netData.NeuralNetworkName = "";
            netData.DNAVersion = 0;
            netData.NeuralNetworkDna.Biases.Clear();
            netData.NeuralNetworkDna.Weights.Clear();
            netData.NotationCoefficient = 0;
            netData.NetworkBestResults.Clear();
        }
        private void Update()
        {
            if (Input.GetKeyDown(KeyCode.A) && !isNeuralNetTraining && !isNeuralNetExecuting)
            {
                if(NetworkMode == ENetworkMode.Train) InitializeTraining(NetWorkPrefab, NetworkMode, TrainingBatchSize, Epochs);
                if(NetworkMode == ENetworkMode.Execute) InitializeExecuting(NetWorkPrefab, NetworkMode);
            }
        }

        public void Train()
        {
            if(!isNeuralNetTraining) InitializeTraining(NetWorkPrefab, NetworkMode, TrainingBatchSize, Epochs);
        }
        public void Execute()
        {
            if(!isNeuralNetExecuting) InitializeExecuting(NetWorkPrefab, NetworkMode);
        }

        void SetManagerData()
        {
            if (TrainMode == ETrainMode.NewTraining)
            {
                ResetNetData(NetData);
            }
            if (TrainMode == ETrainMode.LoadFromJson)
            {
                if (LoadFromJson)
                {
                    NetData = LoadNetData(SaveNetDataFileName);
                    Debug.Log("Loaded from Json");
                    ActualBestDNA.Results = NetData.NetworkBestResults;
                    ActualBestDNA.InstanceWeights = NetData.NeuralNetworkDna.Weights;
                    ActualBestDNA.InstanceBiases = NetData.NeuralNetworkDna.Biases;
                    ActualBestDNA.NotationCoefficient = NetData.NotationCoefficient;
                    TrainingRate = NetData.NetworkTrainingRate;
                    TrainingBestResults = new double[ActualBestDNA.Results.Count];
                    for (int j = 0; j < ActualBestDNA.Results.Count; j++)
                    {
                        TrainingBestResults[j] = ActualBestDNA.Results[j];
                    }
                }
                if (!LoadFromJson)
                {
                    Debug.Log("Load From Json = false.");
                }
                
            }
        }
        private void InitializeTraining(GameObject networkPrefab, ENetworkMode eNetworkMode, int batchSize, int epochs)
        {
            isNeuralNetTraining = true;
            isNeuralNetExecuting = false;
            int _instanceID = 0;
            SetManagerData();
            for (int i = 0; i < batchSize; i++)
            {
                GameObject _instanceNet = Instantiate(networkPrefab, this.transform);
                NeuralNet _instance = _instanceNet.GetComponent<NeuralNet>();
                NeuralNetworkInstances.Add(_instance);
                _instance.InitializeNeuralNetwork(eNetworkMode, epochs, _instanceID, this, NetData);
               _instanceID++;
            }
            if (NewTraining)
            {
                if (NetworkFunction == ENetworkFunction.ComputeData)
                {
                    TrainingBestResults = new double[NeuralNetworkInstances[0].OutputLayer.NeuronsInLayer.Count];
                    //InternalParameters = new double[NeuralNetworkInstances[0].OutputLayer.NeuronsInLayer.Count];
                }

                if (NetworkFunction == ENetworkFunction.ControlEntity)
                {
                    TrainingBestResults = new double[NeuralNetworkInstances[0].NeuralNetworkComponent.Controller.EvaluationParameters.Count];
                }
            }
           
        }

        void InitializeExecuting(GameObject networkPrefab, ENetworkMode eNetworkMode)
        {
            isNeuralNetTraining = false;
            isNeuralNetExecuting = true;
            SetManagerData();
            GameObject _instanceNet = Instantiate(networkPrefab, this.transform);
            NeuralNet _instance = _instanceNet.GetComponent<NeuralNet>();
            NeuralNetworkInstances.Add(_instance);
            _instance.InitializeNeuralNetwork(eNetworkMode, 1, 0, this, NetData);
        }

        public void OnInstanceHasEnd(NeuralNet instanceNetwork, double computedCoeff, List<double> result, List<double> allCoeffs)
        {
            InstancesEndedCount++;
            NeuralNetworkEvaluate _instanceNetworkEvaluate = new NeuralNetworkEvaluate(); 
            NeuralNet.DNA dna = new NeuralNet.DNA();
            dna = GetInstanceDNA(instanceNetwork);
            _instanceNetworkEvaluate.InstanceWeights = new List<double>();
            _instanceNetworkEvaluate.InstanceBiases = new List<double>();
            _instanceNetworkEvaluate.InstanceWeights = dna.Weights;
            _instanceNetworkEvaluate.InstanceBiases = dna.Biases;
            _instanceNetworkEvaluate.Results = new List<double>();
            foreach (var res in result)
            {
                _instanceNetworkEvaluate.Results.Add(res);
            }
            _instanceNetworkEvaluate.NotationCoefficient = computedCoeff;
            CompareDNAsAndSaveBest(_instanceNetworkEvaluate, instanceNetwork);
        }
        void CompareDNAsAndSaveBest(NeuralNetworkEvaluate evaluatedDNA, NeuralNet instanceNetwork)
        {
            EvaluateAllInstancesForIteration(evaluatedDNA, ActualBestDNA);
        }

        private void EvaluateAllInstancesForIteration(NeuralNetworkEvaluate evaluationInstance, NeuralNetworkEvaluate ActualDna)
        {
            EvaluateDNAForInstancesIteration.Add(evaluationInstance);
            if (ActualBestDNA.InstanceWeights.Count == 0)
            {
                if (InstancesEndedCount >= NeuralNetworkInstances.Count-1)
                {
                    var best = ReturnBestCoefficientNetworkForThisIteration(EvaluateDNAForInstancesIteration, evaluateIterationValueToActualDna);
                    ActualBestDNA = best;
                    HandleAndDisplayResults(best.Results);
                    SaveNetData(best.InstanceWeights, best.InstanceBiases, best.Results, best.NotationCoefficient);
                    DNAVersion = NetData.DNAVersion;
                    EvaluateDNAForInstancesIteration.Clear();
                    DNAHasUpgrade = true;
                    Debug.Log("DNA Version has been upgraded");
                    StartNextEpoch();
                    
                } 
               
            }
            else
            {
                if (InstancesEndedCount >= NeuralNetworkInstances.Count-1)
                {
                    var best = ReturnBestCoefficientNetworkForThisIteration(EvaluateDNAForInstancesIteration, evaluateIterationValueToActualDna);
                    if (evaluateIterationValueToActualDna == EEvaluateIterationValueToActualDNA.IterationResultsInferior)
                    {
                        if (best.NotationCoefficient < ActualBestDNA.NotationCoefficient)
                        {
                            DNAHasUpgrade = true;
                            ManageTrainingRateOnFeedback(DNAHasUpgrade, best.NotationCoefficient, ActualBestDNA.NotationCoefficient);
                            ActualBestDNA = best;
                            HandleAndDisplayResults(best.Results);
                            SaveNetData(best.InstanceWeights, best.InstanceBiases, best.Results, best.NotationCoefficient);
                            DNAVersion = NetData.DNAVersion;
                            EvaluateDNAForInstancesIteration.Clear();
                            
                            Debug.Log("DNA Version has been upgraded");
                        }
                        else
                        {
                            ManageTrainingRateOnFeedback(DNAHasUpgrade, best.NotationCoefficient, ActualDna.NotationCoefficient);
                        }
                    }
                    
                    if (evaluateIterationValueToActualDna == EEvaluateIterationValueToActualDNA.IterationResultsSuperior)
                    {
                        if (best.NotationCoefficient > ActualBestDNA.NotationCoefficient)
                        {
                            DNAHasUpgrade = true;
                            ManageTrainingRateOnFeedback(DNAHasUpgrade, best.NotationCoefficient, ActualBestDNA.NotationCoefficient);
                            Debug.Log(best.NotationCoefficient + " best and actual : " + ActualBestDNA.NotationCoefficient);
                            ActualBestDNA = best;
                            HandleAndDisplayResults(best.Results);
                            SaveNetData(best.InstanceWeights, best.InstanceBiases, best.Results, best.NotationCoefficient);
                            DNAVersion = NetData.DNAVersion;
                            EvaluateDNAForInstancesIteration.Clear();
                            
                            Debug.Log("DNA Version has been upgraded");
                        }
                        else
                        {
                            Debug.Log(best.NotationCoefficient + " best and actual : " + ActualBestDNA.NotationCoefficient);
                            ManageTrainingRateOnFeedback(DNAHasUpgrade, best.NotationCoefficient, ActualDna.NotationCoefficient);
                        }
                    }
                    
                    StartNextEpoch();
                } 
            }
        }

        public void BypassTrainingFeedBackEvaluationAndStartNextEpoch()
        {
            StartNextEpoch();
        }
        private NeuralNetworkEvaluate ReturnBestCoefficientNetworkForThisIteration(List<NeuralNetworkEvaluate> entry, EEvaluateIterationValueToActualDNA bestValueToActualDnaType)
        {
            entry.Sort(delegate(NeuralNetworkEvaluate A, NeuralNetworkEvaluate B)
            {
                return A.NotationCoefficient.CompareTo(B.NotationCoefficient); 
            });
            NeuralNetworkEvaluate toReturn = new NeuralNetworkEvaluate();
            if (bestValueToActualDnaType == EEvaluateIterationValueToActualDNA.IterationResultsSuperior)
            {
                if (entry[0].NotationCoefficient > entry[entry.Count - 1].NotationCoefficient)
                {
                    toReturn = entry[0];
                }
                else
                {
                    toReturn = entry[entry.Count - 1];
                }
            }

            if (bestValueToActualDnaType == EEvaluateIterationValueToActualDNA.IterationResultsInferior)
            {
                if (entry[0].NotationCoefficient < entry[entry.Count - 1].NotationCoefficient)
                {
                    toReturn = entry[0];
                }
                else
                {
                    toReturn = entry[entry.Count - 1];
                }
            }
           
            return toReturn;
        }

        private NeuralNet.DNA GetInstanceDNA(NeuralNet instanceNetwork)
        {
            NeuralNet.DNA newDNA = new NeuralNet.DNA();
            newDNA.Weights = new List<double>();
            newDNA.Biases = new List<double>();

            List<double> biases = new List<double>();
            List<double> weights = new List<double>();
            for (int i = 0; i < instanceNetwork.InputLayer.NeuronsInLayer.Count; i++)
            {
                weights.Add(instanceNetwork.InputLayer.NeuronsInLayer[i].Weights[0]);
            }
            biases.Add(instanceNetwork.InputLayer.LayerBias);
            for (int i = 0; i < instanceNetwork.HiddenLayers.Count; i++)
            {
                for (int j = 0; j < instanceNetwork.HiddenLayers[i].NeuronsInLayer.Count; j++)
                {
                    for (int k = 0; k < instanceNetwork.HiddenLayers[i].NeuronsInLayer[j].PreviousNeurons; k++)
                    {
                        weights.Add(instanceNetwork.HiddenLayers[i].NeuronsInLayer[j].Weights[k]);
                    }
                }
                biases.Add(instanceNetwork.HiddenLayers[i].LayerBias);
            }
            
            for (int j = 0; j < instanceNetwork.OutputLayer.NeuronsInLayer.Count; j++)
            {
                 for (int k = 0; k < instanceNetwork.OutputLayer.NeuronsInLayer[j].PreviousNeurons; k++)
                 {
                     weights.Add(instanceNetwork.OutputLayer.NeuronsInLayer[j].Weights[k]);
                 }
            }
            biases.Add(instanceNetwork.OutputLayer.LayerBias);
            newDNA.Biases = biases;
            newDNA.Weights = weights;
            return newDNA;
        }

        
        private void StartNextEpoch()
        {
            if (isNeuralNetTraining)
            {
                EpochsCount++;
                if (EpochsCount >= Epochs)
                {
                    isNeuralNetTraining = false;
                    foreach (var instanceNet in NeuralNetworkInstances)
                    {
                        instanceNet.IsTraining = false;
                        instanceNet.IsExecuting = false;
                        EpochsCount = 0;
                    }
                    Debug.Log("ENDING TRAINING");
                }

                if (InstancesEndedCount >= NeuralNetworkInstances.Count - 1)
                {
                    RestartInstances(NeuralNetworkInstances);
                }
            }

            if (isNeuralNetExecuting)
            {
                NeuralNetworkInstances[0].RestartInstance(NetworkMode, NetData, DNAHasUpgrade, ForceInstanceDNAReset);
            }
        }
      
        private void RestartInstances(List<NeuralNet> neuralNets)
        {
            foreach (var netInstance in neuralNets)
            {
                netInstance.RestartInstance(NetworkMode, NetData, DNAHasUpgrade, ForceInstanceDNAReset);
            }
            EvaluateDNAForInstancesIteration.Clear();
            ForceInstanceDNAReset = false;
            DNAHasUpgrade = false;
            InstancesEndedCount = 0;
        }
        private void ManageTrainingRateOnFeedback(bool dnaHasUpgrade, double bestCoeff = 0, double actualCoeff = 0)
        {
            if (AdjustTrainingRateAutomatically)
            {
                if (!dnaHasUpgrade)
                {
                    epochsWithoutDNAEvolutionCount++;
                    previousIterationsCoefficients.Add(bestCoeff);
                    previousIterationsCoefficientAverage =
                        NeuralMathCompute.AverageFromList(previousIterationsCoefficients);
                    if (epochsWithoutDNAEvolutionCount > TrainingRateEvolution)
                    {
                        ForceInstanceDNAReset = true;
                        if (evaluateIterationValueToActualDna ==
                            EEvaluateIterationValueToActualDNA.IterationResultsSuperior)
                        {
                            double averageCoefficient = 0;
                            double actualTrainingRate = TrainingRate;
                            for (int i = 0; i < previousIterationsCoefficients.Count; i++)
                            {
                                averageCoefficient += previousIterationsCoefficients[i];
                            }
                            averageCoefficient /= previousIterationsCoefficients.Count;
                            double downgradeDelta = ((actualCoeff - averageCoefficient)/averageCoefficient)*TrainingRateChangePurcentage/100;
                            Debug.Log(averageCoefficient + " > average coeff, " + actualTrainingRate + " actual, " + downgradeDelta + " > downgradeRatio.");
                            actualTrainingRate += downgradeDelta;
                            TrainingRate = actualTrainingRate;
                            TrainingRate = Mathf.Clamp((float)TrainingRate, 0.000001f, (float)MaxTrainingRate);
                        }
                        else
                        {
                            previousIterationsCoefficients.Clear();
                            previousIterationsCoefficientAverage = 0;
                            double actualTrainingRate = TrainingRate;
                            Debug.Log("actualTrainingRate" + actualTrainingRate + " best" + bestCoeff + " actualcoeff" + actualCoeff);
                            double upgradeDelta = ((bestCoeff - actualCoeff) / bestCoeff) * TrainingRateChangePurcentage / 100;
                            actualTrainingRate -= upgradeDelta;
                            TrainingRate = actualTrainingRate;
                            TrainingRate = Mathf.Clamp((float)TrainingRate, 0.000001f, (float)MaxTrainingRate);
                            Debug.Log("Training Rate Decreased");
                        }
                        
                        epochsWithoutDNAEvolutionCount = 0;
                        Debug.Log("Training Rate Increased");

                    }
                }

                if (dnaHasUpgrade)
                {
                    if (evaluateIterationValueToActualDna ==
                        EEvaluateIterationValueToActualDNA.IterationResultsSuperior)
                    {
                        previousIterationsCoefficients.Clear();
                        previousIterationsCoefficientAverage = 0;
                        double actualTrainingRate = TrainingRate;
                        Debug.Log("actualTrainingRate" + actualTrainingRate + " best" + bestCoeff + " actualcoeff" + actualCoeff);
                        double upgradeDelta = ((bestCoeff - actualCoeff) / bestCoeff) * TrainingRateChangePurcentage / 100;
                        actualTrainingRate -= upgradeDelta;
                        TrainingRate = actualTrainingRate;
                        TrainingRate = Mathf.Clamp((float)TrainingRate, 0.000001f, (float)MaxTrainingRate);
                        Debug.Log("Training Rate Decreased");
                    }
                    else
                    {
                        double averageCoefficient = 0;
                        double actualTrainingRate = TrainingRate;
                        for (int i = 0; i < previousIterationsCoefficients.Count; i++)
                        {
                            averageCoefficient += previousIterationsCoefficients[i];
                        }
                        averageCoefficient /= previousIterationsCoefficients.Count;
                        double downgradeDelta = ((actualCoeff - averageCoefficient)/averageCoefficient)*TrainingRateChangePurcentage/100;
                        Debug.Log(averageCoefficient + " > average coeff, " + actualTrainingRate + " actual, " + downgradeDelta + " > downgradeRatio.");
                        actualTrainingRate += downgradeDelta;
                        TrainingRate = actualTrainingRate;
                        TrainingRate = Mathf.Clamp((float)TrainingRate, 0.000001f, (float)MaxTrainingRate);
                    }
                }
            }
            
            
        }
        
        private void HandleAndDisplayResults(List<double> result)
        {
            for (int i = 0; i < result.Count; i++)
            {
                TrainingBestResults[i] = result[i];
            }
        }
        
        private void SaveNetData(List<double> instanceWeights, List<double> instanceBiases, List<double> results, double notationCoefficient)
        {
            if (NewTraining)
            {
                NetData.StartTrainingRate = TrainingRate;
            }
            NetData.NeuralNetworkDna = new NeuralNet.DNA();
            NetData.NeuralNetworkDna.Weights = instanceWeights;
            NetData.NeuralNetworkDna.Biases = instanceBiases;
            NetData.NetworkBestResults = results;
            NetData.NotationCoefficient = notationCoefficient;
            NetData.NeuralNetworkName = IANetworkName;
            NetData.NetworkTrainingRate = TrainingRate;
            NetData.DNAVersion++;
            NetData.HasData = true;
            NewTraining = false;
            // Serialize
            NeuralNetworkSerializer.Save(NetData, SaveNetDataFileName);
        }

        private NetData LoadNetData(string fileName)
        {
            Scripts.Data.NetData loadedData = new NetData();
            loadedData = NeuralNetworkSerializer.Load(loadedData, fileName);
            return loadedData;
        }


        public NeuralNetworkBluePrint CreateBlueprint(NeuralNet neuralNet, string blueprintFileName)
        {
            NeuralNetworkBluePrint newBluePrint = new NeuralNetworkBluePrint();
            newBluePrint.InputLayerConstruct = neuralNet.InputLayerConstruct;
            newBluePrint.HiddenLayerConstruct = neuralNet.HiddenLayerConstruct;
            newBluePrint.OutputLayerConstruct = neuralNet.OutputLayerConstruct;
            newBluePrint.NetData = neuralNet._NetData;
            return newBluePrint;
        }
        
        
        private void SaveBlueprint(NeuralNet neuralNet, string blueprintFileName, NeuralNetworkBluePrint saveBluePrint)
        {
            NeuralNetworkBluePrint saveBlueprint = new NeuralNetworkBluePrint();
            saveBlueprint = LoadBlueprint(blueprintFileName);
            saveBlueprint.InputLayerConstruct = neuralNet.InputLayerConstruct;
            saveBlueprint.HiddenLayerConstruct = neuralNet.HiddenLayerConstruct;
            saveBlueprint.OutputLayerConstruct = neuralNet.OutputLayerConstruct;
            saveBlueprint.NetData = neuralNet._NetData;
            NeuralNetworkSerializer.GenericSave(saveBlueprint, blueprintFileName);

        }

        private NeuralNetworkBluePrint LoadBlueprint(string blueprintFileName)
        {
            NeuralNetworkBluePrint loadedBlueprint = new NeuralNetworkBluePrint();
            loadedBlueprint = NeuralNetworkSerializer.GenericLoad(loadedBlueprint, blueprintFileName);
            return loadedBlueprint;
        }

        public NeuralNet CreateNeuralNetworkFromBlueprint(string BlueprintFileName)
        {
            GameObject newNetwork = Instantiate(new GameObject(), transform);
            NeuralNet neuralNet =  newNetwork.AddComponent<NeuralNet>();
            NeuralNetworkBluePrint networkBluePrint = LoadBlueprint(BlueprintFileName);
            neuralNet.InputLayerConstruct = networkBluePrint.InputLayerConstruct;
            neuralNet.HiddenLayerConstruct = networkBluePrint.HiddenLayerConstruct;
            neuralNet.OutputLayerConstruct = networkBluePrint.OutputLayerConstruct;
            neuralNet._NetData = networkBluePrint.NetData;
            return neuralNet;
        }
        
    }
}