using System;
using UnityEngine;

namespace NeuralNetwork.Scripts.Data
{
    [Serializable]
    public class NeuralNetworkPerformanceSolver
    {
        public string ParameterName;
        public double EvaluationParameter; // result at the InstanceEnd 
        [Range(0.05f, 10f)] public double EvaluationParameterWeight = 1; // to modify the importance of the parameter in the computed PerformanceCoefficient.
        public EParameterType ParameterType; // how it should be comparate to other results

        public double ExpectedValue; // Refactoring of NeuralNetworkManager.InternalParameters
    }

    public enum EParameterType
    {
        ShouldBeInferior, //for the first epoch of a training, needs init value
        ShouldBeSuperior,
        ConvergeToExpectedValue,
    }

    
}