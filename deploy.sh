#!/bin/bash

kubectl apply -f k8s/jobs/ns-pvc.yaml

kubectl apply -f k8s/jobs/rbac.yaml

kubectl apply -f k8s/jobs/job1-collect.yaml
kubectl wait -n energy --for=condition=complete --timeout=1m job/collect-snapshot

kubectl apply -f k8s/jobs/job2-label.yaml
kubectl wait -n energy --for=condition=complete --timeout=30m job/build-labels

kubectl apply -f k8s/jobs/job3-dataset.yaml
kubectl wait -n energy --for=condition=complete --timeout=30m job/join-train-rows

kubectl apply -f k8s/jobs/job4-train.yaml
kubectl wait -n energy --for=condition=complete --timeout=60m job/train-power

kubectl apply -f k8s/deploy-podpower-predict.yaml
kubectl wait -n energy --for=condition=available --timeout=30m deployment/podpower-predict

kubectl apply -f k8s/deploy-podpower-collector.yaml
kubectl wait -n energy --for=condition=available --timeout=30m deployment/podpower-collector
