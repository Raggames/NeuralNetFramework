using System.Collections.Generic;
using UnityEngine;

namespace NeuralNetwork
{
    [RequireComponent(typeof(NeuralNet))]
    public class NeuralNetworkComponent : MonoBehaviour
    {
        public NeuralNet NeuralNet;
        public List<NetInput> NetInput; //use a net input to connect a dynamic value to his NeuralNetInstance 
        public List<NetOutput> NetOutput;

        public NeuralNetworkController Controller;
        public bool inputStreamOn;
        private void Start()
        {
            NeuralNet = GetComponent<NeuralNet>();
           
            if (NetOutput.Count == 0)
            {
                for (int i = 0; i <  NeuralNet.OutputLayerConstruct[0].NeuronCount; i++)
                {
                    NetOutput.Add(new NetOutput()
                    {
                        OutputName = "Output" + i,
                        OutputValue = 0,
                    });
                }
            }
            if (NetInput.Count == 0)
            {
                for (int i = 0; i <  NeuralNet.InputLayerConstruct[0].NeuronCount; i++)
                {
                    NetInput.Add(new NetInput()
                    {
                        InputName = "Input" + i,
                        InputValue = 0,
                    });
                }
            }
        }

        public void Restart()
        {
            Controller.InstanceReset();
        }
        
        void InputToNeuralNet(NeuralNet neuralNet)
        {
            if (NeuralNet.IsTraining || NeuralNet.IsExecuting)
            {
                inputStreamOn = true;
                for (int i = 0; i < NeuralNet.InputLayer.NeuronsInLayer.Count; i++)
                {
                    NeuralNet.InputLayer.NeuronsInLayer[i].Inputs.Clear();
                    NeuralNet.InputLayer.NeuronsInLayer[i].ReceiveInput(NetInput[i].InputValue);
                }
                
            }
            else
            {
                inputStreamOn = false;
            }
        }

        public void InstanceEnd(List<double> parameters, NeuralNetworkController controller)
        {
            this.NeuralNet.OnInstanceEnd(parameters, controller);
        }

        private void Update()
        {
            InputToNeuralNet(NeuralNet);
        }
    }
}