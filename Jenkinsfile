pipeline {
    agent any

    environment {
        DOCKER_HUB_REPO = 'bapurolling/end-to-end'
        IMAGE_TAG = "${env.BUILD_ID}" 
        DOCKER_HUB_CREDENTIALS = 'dockerhub_credentials' 
        GIT_REPO = 'https://github.com/Bapurolling/Text_Emotion_Detector.git'
    }

    triggers {
        githubPush()  
    }

    stages {
        stage('Install Git LFS') {
            steps {
                script {
                    // Install Git LFS
                    sh '''
                    sudo apt-get update
                    sudo apt-get install -y git-lfs
                    git lfs install
                    '''
                }
            }
        }

        stage('Clone Repository') {
            steps {
                script {
                    // Clone the repository with Git LFS support
                    sh '''
                    git clone ${GIT_REPO}
                    cd Text_Emotion_Detector
                    git lfs pull
                    '''
                }
            }
        }

        stage('Build Docker Image') {
            steps {
                script {
                    // Build the Docker image from the Dockerfile
                    sh "docker build -t ${DOCKER_HUB_REPO}:${IMAGE_TAG} ."
                }
            }
        }

        stage('Push Docker Image') {
            steps {
                script {
                    // Log in to Docker Hub
                    withCredentials([usernamePassword(credentialsId: DOCKER_HUB_CREDENTIALS, usernameVariable: 'DOCKER_USER', passwordVariable: 'DOCKER_PASSWORD')]) {
                        sh "echo $DOCKER_PASSWORD | docker login -u $DOCKER_USER --password-stdin"
                    }
                    // Push the image to Docker Hub
                    sh "docker push ${DOCKER_HUB_REPO}:${IMAGE_TAG}"
                }
            }
        }

        stage('Pull Docker Image') {
            steps {
                script {
                    // Pull the latest Docker image from Docker Hub
                    sh "docker pull ${DOCKER_HUB_REPO}:${IMAGE_TAG}"
                }
            }
        }

        stage('Stop Existing Container') {
            steps {
                script {
                    // Stop and remove the existing container if it is running
                    sh '''
                    if [ $(docker ps -q -f name=text_emotion_detector) ]; then
                        docker stop text_emotion_detector
                        docker rm text_emotion_detector
                    fi
                    '''
                }
            }
        }

        stage('Run New Container') {
            steps {
                script {
                    // Run the new container from the pulled image
                    sh "docker run -d -p 8501:8501 --name text_emotion_detector ${DOCKER_HUB_REPO}:${IMAGE_TAG}"
                }
            }
        }
    }

    post {
        always {
            cleanWs()
        }
    }
}
