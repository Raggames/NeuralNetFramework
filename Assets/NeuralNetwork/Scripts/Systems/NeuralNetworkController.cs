using System.Collections;
using System.Collections.Generic;
using NeuralNetwork;
using NeuralNetwork.Scripts.Data;
using UnityEngine;

public abstract class NeuralNetworkController : MonoBehaviour
{
    public NeuralNetworkComponent NeuralNetworkComponent;

    public List<NeuralNetworkPerformanceSolver> EvaluationParameters = new List<NeuralNetworkPerformanceSolver>();
    
        public abstract void SetInputs();
        public abstract void OnOutput();
        public abstract void OnInstanceFail();
        public abstract void InstanceReset();


}



