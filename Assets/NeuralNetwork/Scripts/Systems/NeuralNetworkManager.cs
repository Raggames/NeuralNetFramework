using System.Collections.Generic;
using System.Globalization;
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
        [HideInInspector] public List<NeuralNetworkEvaluate> EvaluateUpgradedsDNAOfEpoch = new List<NeuralNetworkEvaluate>();
        public List<NeuralNetworkEvaluate> KeepAllInstancesDNAOfEpoch = new List<NeuralNetworkEvaluate>();
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
       

        
        [HideInInspector] public bool NewTraining = true;//check if instances is new or has already iterated
        [Header("Training Setup")] 
        public bool LoadFromJson;//will be removed after saving architecture refactoring
        public ETrainMode TrainMode;
        public enum ETrainMode
        {
            NewTraining,
            LoadFromJson,
        }

        public ESaveMode SaveFile;
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
        private bool DNAShouldUpgrade;
        private bool ForceInstanceDNAReset; //if iteration don't get any upgrade avec n = TrainingRateEvolution epochs, reset on DNA data
        private int epochsWithoutDNAEvolutionCount;
        private List<double> previousIterationsCoefficients = new List<double>();
        [SerializeField] private double previousIterationsCoefficientAverage;
        
        public int EpochsCount;
        
        [Header("Neural Network Evaluation")] 
        public int DNAVersion;
        public double[] TrainingBestResults;
        
        #endregion

        private void Start()
        {
            if (TrainMode == ETrainMode.NewTraining)
            {
                NewTraining = true;
            }
            else
            {
                NewTraining = false;
            }
        }

        private void Update()
        {
            if (Input.GetKeyDown(KeyCode.A) && !isNeuralNetTraining && !isNeuralNetExecuting)
            {
                if(NetworkMode == ENetworkMode.Train) InitializeTraining(NetWorkPrefab, NetworkMode, TrainingBatchSize, Epochs);
                if(NetworkMode == ENetworkMode.Execute) InitializeExecuting(NetWorkPrefab, NetworkMode, TrainingBatchSize);
            }
        }
        
        private void ResetNetData(NetData netData)
        {
            netData.HasData = false;
            netData.NeuralNetworkName = "";
            netData.DNAVersion = 0;
            netData.NeuralNetworkDna.Biases.Clear();
            netData.NeuralNetworkDna.Weights.Clear();
            netData.PerformanceCoefficient = 0;
            netData.PerformanceSolvers.Clear();
        }
        
        public void Train()
        {
            if(!isNeuralNetTraining) InitializeTraining(NetWorkPrefab, NetworkMode, TrainingBatchSize, Epochs);
        }
        public void Execute()
        {
            if(!isNeuralNetExecuting) InitializeExecuting(NetWorkPrefab, NetworkMode, TrainingBatchSize);
        }

        #region Initializing
        private void SetManagerData()
        {
            if (TrainMode == ETrainMode.NewTraining)
            {
                if (File.Exists(Application.dataPath + "/StreamingAssets/" + SaveNetDataFileName) && SaveFile == ESaveMode.Override)
                {
                    Debug.Log("An Existing save has been found, you should change the Save File Name if you don't wan't to override or set SaveMode to new file");
                    
                }
                if (File.Exists(Application.dataPath + "/StreamingAssets/" + SaveNetDataFileName) && SaveFile == ESaveMode.New)
                {
                    Debug.Log("An Existing save has been found, setting File Name To New Name");
                    Scripts.Data.NetData netData = LoadNetData(SaveNetDataFileName);
                    SaveNetDataFileName = netData.NeuralNetworkName + "_newAutoSavedFiled";
                    ResetNetData(NetData);
                }

                else
                {
                    ResetNetData(NetData);
                }
                
            }
            if (TrainMode == ETrainMode.LoadFromJson)
            {
                if (LoadFromJson)
                {
                    NetData = LoadNetData(SaveNetDataFileName);
                    Debug.Log("Loaded from Json");
                    NewTraining = NetData.NewTraining;
                    ActualBestDNA.PerformanceSolvers = NetData.PerformanceSolvers;
                    ActualBestDNA.InstanceWeights = NetData.NeuralNetworkDna.Weights;
                    ActualBestDNA.InstanceBiases = NetData.NeuralNetworkDna.Biases;
                    ActualBestDNA.PerformanceCoefficient = NetData.PerformanceCoefficient;
                    TrainingRate = NetData.NetworkTrainingRate;
                    TrainingBestResults = new double[ActualBestDNA.PerformanceSolvers.Count];
                    for (int j = 0; j < ActualBestDNA.PerformanceSolvers.Count; j++)
                    {
                        TrainingBestResults[j] = ActualBestDNA.PerformanceSolvers[j].EvaluationParameter;
                    }
                }
                if (!LoadFromJson)
                {
                    Debug.Log("Save Security Alert. Load From Json = false. Set to New training and restart");
                }
                
            }
        }
        private void ForceSetManagerData()
        {
            NetData = LoadNetData(SaveNetDataFileName);
            Debug.Log("Loaded from Json");
            ActualBestDNA.PerformanceSolvers = NetData.PerformanceSolvers;
            ActualBestDNA.InstanceWeights = NetData.NeuralNetworkDna.Weights;
            ActualBestDNA.InstanceBiases = NetData.NeuralNetworkDna.Biases;
            ActualBestDNA.PerformanceCoefficient = NetData.PerformanceCoefficient;
            TrainingRate = NetData.NetworkTrainingRate;
            TrainingBestResults = new double[ActualBestDNA.PerformanceSolvers.Count];
            for (int j = 0; j < ActualBestDNA.PerformanceSolvers.Count; j++)
            {
                TrainingBestResults[j] = ActualBestDNA.PerformanceSolvers[j].EvaluationParameter;
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

        private void InitializeExecuting(GameObject networkPrefab, ENetworkMode eNetworkMode, int batchSize)
        {
            isNeuralNetTraining = false;
            isNeuralNetExecuting = true;
            int _instanceID = 0;
            ForceSetManagerData();
            for (int i = 0; i < batchSize; i++)
            {
                GameObject _instanceNet = Instantiate(networkPrefab, this.transform);
                NeuralNet _instance = _instanceNet.GetComponent<NeuralNet>();
                NeuralNetworkInstances.Add(_instance);
                _instance.InitializeNeuralNetwork(eNetworkMode, 1, _instanceID, this, NetData);
                _instanceID++;
            }
            
        }

        #endregion
        #region Evaluating
        public void OnInstanceHasEnd(NeuralNet instanceNetwork, double computedCoeff, List<NeuralNetworkPerformanceSolver> solvers, bool instancehasBestDna)
        {
            InstancesEndedCount++;
            NeuralNetworkEvaluate _instanceNetworkEvaluate = new NeuralNetworkEvaluate();
                NeuralNet.DNA dna = new NeuralNet.DNA();
                dna = GetInstanceDNA(instanceNetwork);
                _instanceNetworkEvaluate.InstanceWeights = new List<double>();
                _instanceNetworkEvaluate.InstanceBiases = new List<double>();
                _instanceNetworkEvaluate.InstanceWeights = dna.Weights;
                _instanceNetworkEvaluate.InstanceBiases = dna.Biases;
                _instanceNetworkEvaluate.PerformanceSolvers = new List<NeuralNetworkPerformanceSolver>();
                _instanceNetworkEvaluate.PerformanceSolvers = solvers;
                _instanceNetworkEvaluate.PerformanceCoefficient = computedCoeff;
                Debug.Log("computed perf : " + computedCoeff);
                if (instancehasBestDna)
                {
                    EvaluateUpgradedsDNAOfEpoch.Add(_instanceNetworkEvaluate);
                    DNAShouldUpgrade = true;
                }
             
                
            if (InstancesEndedCount == NeuralNetworkInstances.Count-1)
            {
                var best = ReturnBestCoefficientNetworkForThisIteration(EvaluateUpgradedsDNAOfEpoch);
                Debug.Log("BestPerf" + best.PerformanceCoefficient);
                if (NetData.HasData == false)
                {
                    ActualBestDNA = best;
                    HandleAndDisplayResults(best.PerformanceSolvers);
                    SaveNetData(best.InstanceWeights, best.InstanceBiases, best.PerformanceSolvers, best.PerformanceCoefficient);
                    DNAVersion = NetData.DNAVersion;
                    Debug.Log("Net Data was empty, actual iteration best DNA was saved.");
                    StartNextEpoch();
                    
                    //end of loop
                }
                if (DNAShouldUpgrade)
                {
                    Debug.Log("BestPerf" + best.PerformanceCoefficient + " DNA Upgrade");
                    ManageTrainingRateOnFeedback(DNAShouldUpgrade, best.PerformanceCoefficient, ActualBestDNA.PerformanceCoefficient);
                    ActualBestDNA = best;
                    HandleAndDisplayResults(best.PerformanceSolvers);
                    SaveNetData(best.InstanceWeights, best.InstanceBiases, best.PerformanceSolvers, best.PerformanceCoefficient);
                    DNAVersion = NetData.DNAVersion;
                }
                if(!DNAShouldUpgrade)
                {
                    Debug.Log(EvaluateUpgradedsDNAOfEpoch.Count + " evaluate dna count");
                    ManageTrainingRateOnFeedback(DNAShouldUpgrade, _instanceNetworkEvaluate.PerformanceCoefficient, ActualBestDNA.PerformanceCoefficient);

                }
                StartNextEpoch();
            }
            
        }
        private NeuralNetworkEvaluate ReturnBestCoefficientNetworkForThisIteration(List<NeuralNetworkEvaluate> evaluateDnaForInstancesIteration)
        {
            evaluateDnaForInstancesIteration.Sort(delegate(NeuralNetworkEvaluate evaluate, NeuralNetworkEvaluate networkEvaluate)
                {
                    return evaluate.PerformanceCoefficient.CompareTo(networkEvaluate.PerformanceCoefficient);
                });
            
            return evaluateDnaForInstancesIteration[evaluateDnaForInstancesIteration.Count - 1];
        }

        public void BypassTrainingFeedBackEvaluationAndStartNextEpoch()
        {
            StartNextEpoch();
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

        #endregion
        
        #region Restarting
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

                if (InstancesEndedCount == NeuralNetworkInstances.Count-1)
                {
                    Debug.Log("StartNextEpoch" + InstancesEndedCount);
                    RestartInstances(NeuralNetworkInstances);

                }
            }
    
            if (isNeuralNetExecuting)
            {
                NeuralNetworkInstances[0].RestartInstance(NetworkMode, NetData, DNAShouldUpgrade, ForceInstanceDNAReset);
            }
        }
      
        private void RestartInstances(List<NeuralNet> neuralNets)
        {
            foreach (var netInstance in neuralNets)
            {
                netInstance.RestartInstance(NetworkMode, NetData, DNAShouldUpgrade, ForceInstanceDNAReset);
            }
            ForceInstanceDNAReset = false;
            DNAShouldUpgrade = false;
            InstancesEndedCount = 0;
            Debug.Log(InstancesEndedCount + "instanceEndedcount");
            EvaluateUpgradedsDNAOfEpoch.Clear();

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
                        double averageCoefficient = 0;
                        double actualTrainingRate = TrainingRate;
                        for (int i = 0; i < previousIterationsCoefficients.Count; i++)
                        {
                            averageCoefficient += previousIterationsCoefficients[i];
                        }
                        averageCoefficient /= previousIterationsCoefficients.Count;
                        double downgradeDelta = ((actualCoeff - averageCoefficient)/averageCoefficient)*TrainingRateChangePurcentage/100;
                        actualTrainingRate += downgradeDelta;
                        Debug.Log("Training Rate Increased from " + TrainingRate + " to "+ actualTrainingRate);
                        TrainingRate = actualTrainingRate;
                        TrainingRate = Mathf.Clamp((float)TrainingRate, 0.000001f, (float)MaxTrainingRate);
                        epochsWithoutDNAEvolutionCount = 0;
                    }
                }
                if (dnaHasUpgrade)
                {
                    previousIterationsCoefficients.Clear();
                    previousIterationsCoefficientAverage = 0;
                    double actualTrainingRate = TrainingRate;
                    double upgradeDelta = Mathf.Abs((float)bestCoeff - (float)actualCoeff) / actualCoeff * TrainingRateChangePurcentage / 100;
                    Debug.Log("upgrade delta " + upgradeDelta);
                    actualTrainingRate -= upgradeDelta;
                    Debug.Log("Training Rate Decreased from " + TrainingRate + " to " + actualTrainingRate);
                    TrainingRate = actualTrainingRate;
                    TrainingRate = Mathf.Clamp((float) TrainingRate, 0.000001f, (float) MaxTrainingRate);
                }
            }

        }
        #endregion
        
        #region DataManaging
        private void HandleAndDisplayResults(List<NeuralNetworkPerformanceSolver> solvers)
        {
            for (int i = 0; i < solvers.Count; i++)
            {
                TrainingBestResults[i] = solvers[i].EvaluationParameter;
            }
        }
        
        private void SaveNetData(List<double> instanceWeights, List<double> instanceBiases, List<NeuralNetworkPerformanceSolver> solvers, double notationCoefficient)
        {
            if (NewTraining)
            {
                NetData.StartTrainingRate = TrainingRate;
                NetData.NewTraining = false;
            }
            NewTraining = false;
            NetData.NeuralNetworkDna = new NeuralNet.DNA();
            NetData.NeuralNetworkDna.Weights = instanceWeights;
            NetData.NeuralNetworkDna.Biases = instanceBiases;
            NetData.PerformanceSolvers = solvers;
            Debug.Log("changenetdata");
            NetData.PerformanceCoefficient = notationCoefficient;
            NetData.NeuralNetworkName = IANetworkName;
            NetData.NetworkTrainingRate = TrainingRate;
            NetData.DNAVersion++;
            NetData.HasData = true;
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
        #endregion
    }
}