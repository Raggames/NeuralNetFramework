using System.Collections;
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


    public virtual void ComputeData()
    {
        
    }
    public virtual void SetInputs()
    {
        
    }
    public virtual void OnOutput()
    {
            
    }

    public virtual void OnInstanceFail()
    {
            
    }

    public virtual void InstanceReset()
    {
            
    }
      


}



