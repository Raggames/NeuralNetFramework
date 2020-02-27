using System.IO;
using NeuralNetwork.Scripts.Data;
using UnityEngine;

namespace NeuralNetwork
{
    public static class NeuralNetworkSerializer
    {
        public static void Save(NetData gameData, string fileName)
        {
            string saveJson = JsonUtility.ToJson(gameData);
            File.WriteAllText(Application.streamingAssetsPath + fileName + ".txt", saveJson);
            Debug.Log("NetData Saved");
        }
        public static NetData Load(NetData netData, string fileName)
        {
            NetData loadedData = new NetData();
            if (File.Exists(Application.streamingAssetsPath + fileName + ".txt"))
            {
                string LoadJson = File.ReadAllText(Application.streamingAssetsPath + fileName + ".txt");
                loadedData = JsonUtility.FromJson<NetData>(LoadJson);

            }
            return loadedData;
        }
        
        public static void GenericSave<T>(T data, string fileName)
        {
            string saveJson = JsonUtility.ToJson(data);
            File.WriteAllText(Application.streamingAssetsPath + fileName + ".txt", saveJson);
            Debug.Log("NetData Saved");
        }
        public static T GenericLoad<T>(T data, string fileName)
        {
            if (File.Exists(Application.streamingAssetsPath + fileName + ".txt"))
            {
                string LoadJson = File.ReadAllText(Application.streamingAssetsPath + fileName + ".txt");
                data = JsonUtility.FromJson<T>(LoadJson);

            }
            return data;
        }
        
    }
}