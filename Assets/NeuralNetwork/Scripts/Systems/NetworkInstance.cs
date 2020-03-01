using UnityEngine;

namespace NeuralNetwork
{
    public class NetworkInstance
    {
        public int WeightsNumber;

        public static Random random;
        
        private double[] inputs;
        
        private double[][] i_hWeights;
        private double[][] ih_hWeights;
        private double[] h_Biases;
        private double[] h_Outputs;

        private double[][] h_o_Weights;
        private double[] o_Biases;

        private double[] outputs;

        // Gradients de Back-Propagation 
        private double[] oGrads; // output gradients for back-propagation
        private double[] hGrads; // hidden gradients for back-propagation

        // Momentums de Back-Propagation
        private double[][] ihPrevWeightsDelta;  
        private double[] hPrevBiasesDelta;
        private double[][] hoPrevWeightsDelta;
        private double[] oPrevBiasesDelta;
    }
}