# !/bin/bash

# Help message
print_help() {
    echo "Usage: notify.sh <message> [options]"
    echo "Send a notification with the specified message."
    echo
    echo "Arguments:"
    echo "  message         The message to send in the notification."
    echo
    echo "Options:"
    echo "  -h, --help      Show this help message"
    echo "  -t, --title     Title of the notification"
}

TITLE="Update"

# Parse command line arguments
if [[ $# -eq 0 ]]; then
    print_help
    return 1
fi

# Get message text
MESSAGE="$1"
shift

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            print_help
            return 0
            ;;
        -t|--title)
            if [[ -n "$2" ]]; then
                TITLE="$2"
                shift 2
            else
                echo "Error: --title requires a value."
                return 1
            fi
            ;;
        *)
            echo "Unknown option: $1"
            print_help
            return 1
            ;;
    esac
done

# Get the token to send the notification
if [ -f .env ]; then
    source .env
else
    echo "No .env file found. Please create one with the necessary environment variables."
    return 1
    fi

# Check that the topic is set
if [ -z "$TOPIC" ]; then
    echo "Error: TOPIC is not set in the .env file."
    return 1
fi

if [ -z "$TOKEN" ]; then
    echo "Error: TOKEN is not set in the .env file."
    return 1
fi

# Send the notification using curl
curl -u ":$TOKEN" -H "Title: $TITLE" -d "$MESSAGE" https://ntfy.sh/$TOPIC