using System;
using System.Collections.Generic;

namespace NeuralNetwork.Scripts.NetToolbox
{
    [Serializable]
    public class NetworkTrainingStatistics
    {
        public List<TrainStatWrapper> TrainingSessionStatistics = new List<TrainStatWrapper>();
        public string Name;
        public void SetStatEntry(int epochIndex, double epochBestLoss, double epochTrainingRate, double accuracy = 0)
        {
            TrainStatWrapper wrap = new TrainStatWrapper();
            wrap.WrappedStats.Add(epochIndex);
            wrap.WrappedStats.Add(epochBestLoss);
            wrap.WrappedStats.Add(epochTrainingRate);
            wrap.WrappedStats.Add(accuracy);
            TrainingSessionStatistics.Add(wrap);
           
        }
    }
        
    [Serializable]
    public class TrainingStatsData
    {
        public string Name;
        public List<TrainStatWrapper> TrainingSessionStatistics = new List<TrainStatWrapper>();
    }
    [Serializable]
    public class TrainStatWrapper
    {
        public List<double> WrappedStats = new List<double>();
        //WrappedStats[0] =  epoch index
        //WrappedStats[1] =  epoch bestLoss
        //WrappedStats[2] =  epoch TrainingRate

    }
}