using System.Collections.Generic;

namespace NeuralNetwork.Scripts.NetToolbox
{
    public class NetworkTrainingStatistics
    {
        public List<double[]> TrainingSessionStatistics = new List<double[]>();
        public string Name;
        public void SetStatEntry(int epochIndex, double epochBestLoss, double epochTrainingRate)
        {
            double[] statEntry = new double[3];
            statEntry[0] = epochIndex;
            statEntry[1] = epochBestLoss;
            statEntry[2] = epochTrainingRate;
            TrainingSessionStatistics.Add(statEntry);
            NeuralNetworkSerializer.GenericSave(TrainingSessionStatistics, Name);
        }
    }
}