﻿using System.Collections;
using System.Collections.Generic;
using NeuralNetwork;
using NeuralNetwork.Scripts.Data;
using NeuralNetwork.Scripts.NetToolbox;
using UnityEngine;
[RequireComponent(typeof(NeuralNet))]
public abstract class NeuralNetController : MonoBehaviour
{
    public NeuralNet NeuralNet;
    public List<NetLossParameter> EvaluationParameters = new List<NetLossParameter>();
    
        public abstract void SetInputs();
        public abstract void OnOutput();
        public abstract void OnInstanceFail();
        public abstract void InstanceReset();


}



