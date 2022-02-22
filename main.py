import discord
from discord.ext import commands
import os
from gamma_exposure import *

discord_bot_token = os.environ['TOKEN']

bot = commands.Bot(command_prefix="~")


@bot.event
async def on_ready():
    print('Discord bot running as {0.user}'.format(bot))


@bot.command()
async def ge(ctx, arg):
    await ctx.send(f"Retrieving naive gamma exposure for {arg}")

    messageId = ctx.message.id
    #try :
    runGammaExposure(arg, messageId)

    file = discord.File(f"render/{messageId}_1.png", filename=f'{messageId}_1.png')
    embed = discord.Embed(color=0xff0000)
    embed = embed.set_image(url=f"attachment://{messageId}_1.png")
    await ctx.send(file=file, embed=embed)
    file2 = discord.File(f"render/{messageId}_2.png", filename=f'{messageId}_2.png')
    embed2 = discord.Embed(color=0xff0000)
    embed2 = embed2.set_image(url=f"attachment://{messageId}_2.png")
    await ctx.send(file=file2, embed=embed2)
    file3 = discord.File(f"render/{messageId}_3.png", filename=f'{messageId}_3.png')
    embed3 = discord.Embed(color=0xff0000)
    embed3 = embed3.set_image(url=f"attachment://{messageId}_3.png")
    await ctx.send(file=file3, embed=embed3)

    if os.path.exists(f"render/{messageId}_1.png"):
        os.remove(f"render/{messageId}_1.png")
    if os.path.exists(f"render/{messageId}_2.png"):
        os.remove(f"render/{messageId}_2.png")
    if os.path.exists(f"render/{messageId}_3.png"):
        os.remove(f"render/{messageId}_3.png")
    #except Exception as e:
    #    await ctx.send(f"An error occurred processing {arg} - {e}")
@bot.event
async def on_message(message):
    if message.author == bot.user:
        return
    await bot.process_commands(message)

bot.run(discord_bot_token)
