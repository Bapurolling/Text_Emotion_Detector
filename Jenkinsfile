pipeline {
    agent any

    environment {
        DOCKER_HUB_REPO = 'bapurolling/end-to-end'
    }

    triggers {
        githubPush()  
    }

    stages {
        stage('Pull Docker Image') {
            steps {
                script {
                    // Pull the latest Docker image from Docker Hub
                    sh 'docker pull ${DOCKER_HUB_REPO}:latest'
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
                    sh 'docker run -d -p 8501:8501 --name text_emotion_detector ${DOCKER_HUB_REPO}:latest'
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
