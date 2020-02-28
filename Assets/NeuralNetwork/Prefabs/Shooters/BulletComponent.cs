using System.Collections;
using System.Collections.Generic;
using NeuralNetwork.Scripts.Controllers;
using UnityEngine;

public class BulletComponent : MonoBehaviour
{
    public ShooterController controller;
    private void OnCollisionEnter(Collision other)
    {
        if (other.gameObject.CompareTag("Tank"))
        {
            controller.Points+=2;
            controller.FireTimer -= 1f;
            other.gameObject.GetComponent<ShooterController>().OnInstanceFail();
            Destroy(gameObject);
        }
        if(other.gameObject.CompareTag("Bullet"))
        {
            //
        }
        else
        {
            controller.Points--;
            Destroy(gameObject);
        }
        
    }
}
